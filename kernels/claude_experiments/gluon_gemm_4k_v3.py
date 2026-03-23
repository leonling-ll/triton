import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from torch.profiler import profile, ProfilerActivity


# Stage 1: double-buffered global prefetch on CDNA3 (gfx942).
#
# v2 (single-buffer) loop ordering per tile k:
#   ds_read k → buffer_load k+1 → MFMA k → ds_write k+1 → [barrier]
#   The compiler inserts s_waitcnt lgkmcnt(0) before MFMA (LDS read latency)
#   and s_waitcnt vmcnt(0) before ds_write (HBM load latency), both blocking.
#
# v3 (double-buffer) loop ordering per tile k:
#   buffer_load k+1   (non-blocking, issued FIRST)
#   ds_read LDS[l]    (different buffer → no barrier with the following ds_write)
#   MFMA k            (hides HBM latency: vmcnt fires here)
#   ds_write LDS[g]   (vmcnt stall hidden; LDS[g] ≠ LDS[l] → no barrier needed)
#
# Result: HBM latency (200-800 cy) overlaps with MFMA (512 cy/tile), removing
# the vmcnt stall that was blocking compute in v2.
@gluon.jit
def v3_pipeline_s1(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    gLoadLayoutA: gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [512 // BLOCK_K, BLOCK_K // 8],
        [4, 1],
        [1, 0],
    )
    gLoadLayoutB: gl.constexpr = gl.BlockedLayout(
        [8, 1],
        [BLOCK_K // 8, 512 // BLOCK_K],
        [1, 4],
        [0, 1],
    )

    sharedLayoutA: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=2, max_phase=4, order=[1, 0]
    )
    sharedLayoutB: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=2, max_phase=4, order=[0, 1]
    )

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))
    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn
    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[2, 2]
    )
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    # Double-buffered LDS: two slots alternate between "being read" and "being written".
    # Because the read and write always target different slots, no barrier is needed
    # between a store and the subsequent load.
    nBuffers: gl.constexpr = 2
    smemA = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M, BLOCK_K], layout=sharedLayoutA
    )
    smemB = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N], layout=sharedLayoutB
    )

    iterMax = gl.cdiv(K, BLOCK_K)
    gl.assume(iterMax > 0)

    # PROLOGUE: load tile 0 into LDS[0], then advance pointers to tile 1.
    vgpr_a = gl.amd.cdna3.buffer_load(
        ptr=a_base, offsets=a_offsets, mask=offs_ak[None, :] < K, other=0.0
    )
    vgpr_b = gl.amd.cdna3.buffer_load(
        ptr=b_base, offsets=b_offsets, mask=offs_bk[:, None] < K, other=0.0
    )
    smemA.index(0).store(vgpr_a)
    smemB.index(0).store(vgpr_b)
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)

    # MAIN LOOP: for tile k in [0, iterMax-2], issue the global load for tile k+1
    # BEFORE the LDS read + MFMA for tile k, so that HBM latency overlaps with compute.
    for k in range(0, iterMax - 1):
        l_idx = k % 2      # LDS slot holding tile k (compute now)
        g_idx = 1 - l_idx  # LDS slot to write tile k+1 into

        # Issue global load for tile k+1 (non-blocking).
        # a_base already points to tile k+1 (advanced at end of previous iteration,
        # or in the prologue for k=0).
        vgpr_a = gl.amd.cdna3.buffer_load(
            ptr=a_base, offsets=a_offsets,
            mask=offs_ak[None, :] < K - (k + 1) * BLOCK_K, other=0.0,
        )
        vgpr_b = gl.amd.cdna3.buffer_load(
            ptr=b_base, offsets=b_offsets,
            mask=offs_bk[:, None] < K - (k + 1) * BLOCK_K, other=0.0,
        )

        # LDS read + MFMA for tile k.
        # Reads from LDS[l_idx]; writes go to LDS[g_idx] (different slot).
        # No barrier needed between this load and the store below.
        a = smemA.index(l_idx).load(layout=dotOpLayoutA)
        b = smemB.index(l_idx).load(layout=dotOpLayoutB)
        acc = gl.amd.cdna3.mfma(a, b, acc)

        # Write tile k+1 into LDS[g_idx].
        # The vmcnt(0) stall for the buffer_load above is hidden behind the MFMA above.
        smemA.index(g_idx).store(vgpr_a)
        smemB.index(g_idx).store(vgpr_b)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    # EPILOGUE: the last tile (iterMax-1) is already in LDS[(iterMax-1) % 2].
    l_idx = (iterMax - 1) % 2
    a = smemA.index(l_idx).load(layout=dotOpLayoutA)
    b = smemB.index(l_idx).load(layout=dotOpLayoutB)
    acc = gl.amd.cdna3.mfma(a, b, acc)

    c = acc.to(a_ptr.dtype.element_ty)
    gStoreLayoutC: gl.constexpr = mfmaLayout
    c = gl.convert_layout(c, layout=gStoreLayoutC)
    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c, mask=c_mask)


def matmul(a, b, c=None):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    num_warps = 4
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN, 1)
    v3_pipeline_s1[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return c


def verify_correctness():
    torch.manual_seed(0)
    M, K, N = 256, 128, 256

    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

    c_ref = torch.mm(a.float(), b.float()).bfloat16()
    c_opt = matmul(a, b)

    max_err = (c_ref - c_opt).abs().max().item()
    assert max_err < 1.0, f"max error {max_err:.4f} exceeds tolerance"
    print(f"Correctness OK, max diff: {max_err:.4f}")


def profile_kernel(M=4096, K=4096, N=4096, warmup=10, steps=50):
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    for _ in range(warmup):
        matmul(a, b)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(steps):
            matmul(a, b)
    torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    verify_correctness()
    profile_kernel()
