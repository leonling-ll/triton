import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity


@gluon.jit
def v2_lds_swizzle(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,  #
):

    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    gLoadLayoutA: gl.constexpr = gl.BlockedLayout(
        [1, 8],  # sizePerThread
        [512 // BLOCK_K, BLOCK_K // 8],  # threadsPerWarp
        [4, 1],  # warpsPerCTA
        [1, 0],  # order
    )

    gLoadLayoutB: gl.constexpr = gl.BlockedLayout(
        [8, 1],  # sizePerThread
        [BLOCK_K // 8, 512 // BLOCK_K],  # threadsPerWarp
        [1, 4],  # warpsPerCTA
        [0, 1],  # order
    )

    # LDS bank-conflict-free swizzle layouts (CDNA3 / gfx942, BLOCK_K=32, bf16/fp16)
    #
    # Derivation for A tile (BLOCK_M×BLOCK_K, order=[1,0], K along dim 1):
    #   vec             = min(kWidth=8 * 16, 128) / 16 = 8
    #   elemsPerBankRow = (32 banks * 32 bits) / 16 = 64  (CDNA3)
    #   innerDimLength  = BLOCK_K = 32
    #   perPhase        = max(1, 64 / 32) = 2
    #   maxPhase        = max(1, min(simdWidth=16 / perPhase=2, 32 / vec=8)) = min(8, 4) = 4
    #   → SwizzledSharedLayout(8, 2, 4, order=[1, 0])
    #
    # Same arithmetic for B tile (order=[0,1], K along dim 0).
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

    # Scalar base pointers
    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    # Constant offset tensors — computed once, reused every iteration
    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[2, 2]
    )

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    # Allocate single-buffer LDS (BLOCK_M*BLOCK_K*2 + BLOCK_K*BLOCK_N*2 bytes)
    smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K], layout=sharedLayoutA)
    smemB = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [BLOCK_K, BLOCK_N], layout=sharedLayoutB)

    # PROLOGUE: load tile 0 into LDS before the main loop
    ga = gl.amd.cdna3.buffer_load(
        ptr=a_base, offsets=a_offsets,
        mask=offs_ak[None, :] < K, other=0.0,
    )
    gb = gl.amd.cdna3.buffer_load(
        ptr=b_base, offsets=b_offsets,
        mask=offs_bk[:, None] < K, other=0.0,
    )
    smemA.store(ga)
    smemB.store(gb)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)

    # MAIN LOOP: read current tile from LDS, prefetch next tile from global,
    # compute MFMA, then write prefetched tile to LDS.
    # The compiler inserts s_barrier between store→load on the same smem buffer.
    max_iter = gl.cdiv(K, BLOCK_K)
    gl.assume(max_iter > 0)
    for k in range(0, max_iter - 1):
        cur_a = smemA.load(layout=dotOpLayoutA)
        a_base += BLOCK_K * stride_ak
        next_a = gl.amd.cdna3.buffer_load(
            ptr=a_base, offsets=a_offsets,
            mask=offs_ak[None, :] < K - (k + 1) * BLOCK_K, other=0.0,
        )

        cur_b = smemB.load(layout=dotOpLayoutB)
        b_base += BLOCK_K * stride_bk
        next_b = gl.amd.cdna3.buffer_load(
            ptr=b_base, offsets=b_offsets,
            mask=offs_bk[:, None] < K - (k + 1) * BLOCK_K, other=0.0,
        )

        acc = gl.amd.cdna3.mfma(cur_a, cur_b, acc)

        smemA.store(next_a)
        smemB.store(next_b)

    # EPILOGUE: consume the last tile already in LDS
    cur_a = smemA.load(layout=dotOpLayoutA)
    cur_b = smemB.load(layout=dotOpLayoutB)
    acc = gl.amd.cdna3.mfma(cur_a, cur_b, acc)

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
    v2_lds_swizzle[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
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
