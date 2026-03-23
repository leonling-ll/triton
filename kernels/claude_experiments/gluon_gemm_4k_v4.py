import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from torch.profiler import profile, ProfilerActivity


# Stage 1 double-buffered global prefetch on CDNA3 (gfx942), with BLOCK_K=64.
#
# Key changes vs v3 (BLOCK_K=32, also double-buffered, regressed):
#
#   BLOCK_K 32 → 64:
#     - MFMA compute per K-tile: 512 cy → ~1024 cy (doubles; better hides HBM latency)
#     - K-loop iterations:      128   →  64        (halves; fewer prologue/epilogue overhead)
#     - HBM bytes loaded per tile: 2× larger, but latency increase is sub-linear
#       (HBM latency ~200–800 cy base + bandwidth term; we hide more of it with 2× MFMA)
#
#   Swizzle layout updated for BLOCK_K=64 (bf16, CDNA3):
#     vec=8, per_phase=1, max_phase=8 (conflict-free for 64-wide K dim):
#       innerDimLength = BLOCK_K = 64
#       perPhase  = max(1, elemsPerBankRow=64 / innerDim=64)    = 1
#       maxPhase  = max(1, min(simdWidth=16 / perPhase=1, 64/8)) = min(16, 8) = 8
#
#   Double-buffered pipeline (same as v3):
#     Issue buffer_load for tile k+1 → ds_read LDS[l] → MFMA k → ds_write LDS[g]
#     With 1024 cy MFMA the vmcnt(0) stall for the buffer_load has a much larger
#     compute window to fire in.
@gluon.jit
def v4_pipeline_s1_bk64(
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

    # Conflict-free swizzle for BLOCK_K=64, bf16, CDNA3:
    #   perPhase=1, maxPhase=8 (derived in module docstring above)
    sharedLayoutA: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[1, 0]
    )
    sharedLayoutB: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[0, 1]
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

    nBuffers: gl.constexpr = 2
    smemA = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M, BLOCK_K], layout=sharedLayoutA
    )
    smemB = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N], layout=sharedLayoutB
    )

    iterMax = gl.cdiv(K, BLOCK_K)
    gl.assume(iterMax > 0)

    # PROLOGUE: load tile 0 into LDS[0], advance pointers to tile 1.
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

    # MAIN LOOP: issue load for k+1 (non-blocking) → ds_read k → MFMA k → ds_write k+1.
    # With BLOCK_K=64 the MFMA block is ~1024 cycles — large enough to hide the
    # buffer_load latency (~200–800 cy) so vmcnt(0) fires inside the MFMA block.
    for k in range(0, iterMax - 1):
        l_idx = k % 2      # LDS slot holding tile k (compute now)
        g_idx = 1 - l_idx  # LDS slot to write tile k+1 into

        # Issue global load for tile k+1 (non-blocking).
        vgpr_a = gl.amd.cdna3.buffer_load(
            ptr=a_base, offsets=a_offsets,
            mask=offs_ak[None, :] < K - (k + 1) * BLOCK_K, other=0.0,
        )
        vgpr_b = gl.amd.cdna3.buffer_load(
            ptr=b_base, offsets=b_offsets,
            mask=offs_bk[:, None] < K - (k + 1) * BLOCK_K, other=0.0,
        )

        # LDS read + MFMA for tile k.
        # Reads LDS[l_idx]; write below targets LDS[g_idx] — no barrier needed.
        a = smemA.index(l_idx).load(layout=dotOpLayoutA)
        b = smemB.index(l_idx).load(layout=dotOpLayoutB)
        acc = gl.amd.cdna3.mfma(a, b, acc)

        # Write tile k+1 into LDS[g_idx].
        # vmcnt(0) for the buffer_load above should fire during the MFMA block above.
        smemA.index(g_idx).store(vgpr_a)
        smemB.index(g_idx).store(vgpr_b)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    # EPILOGUE: last tile is already in LDS[(iterMax-1) % 2].
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
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    num_warps = 4
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN, 1)
    v4_pipeline_s1_bk64[grid](
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
