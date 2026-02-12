import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import numpy as np
import math
import torch.nn.functional as F

from torch.profiler import profile, record_function, ProfilerActivity


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    # pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    return pid


@triton.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: tl.constexpr = 1):
    """
    Maps 1D pid to 2D grid coords (pid_m, pid_n).

    Args:
        - pid: 1D pid
        - num_pid_m: grid m size
        - num_pid_n: grid n size
        - GROUP_SIZE_M: tl.constexpr: default is 1
    """
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n


# BLOCK_S比较大 效率非常高 在序列长度大于100 BatchSize=80 [3072 768]线性投影中
# 即便真实序列长度等于最大序列长度 与原生实现速度几乎持平
# 适用于序列长度比较大 >= 512的场景
# 如果序列最大长度在[256 ~ 512]之间 BLOCK_S可以选取64
if torch.version.hip:
    autotune_configs = [
        triton.Config({"TILE_D": 32, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8,
                      'matrix_instr_nonkdim': 16, "waves_per_eu": 2, "kpack": 2}, num_warps=4, num_stages=2),
    ]
else:
    autotune_configs = [
        triton.Config({"TILE_D": 64, "BLOCK_S": 64, "BLOCK_D": 128,
                      "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
    ]
# TILE_D: K, BLOCK_S:M, BLOCK_D:N


@triton.autotune(
    configs=autotune_configs,
    key=["M", "D_IN", "D_OUT"],
)
@gluon.jit
def _addmm_x_BSD_variable_length_tile_d_swizzle_long_sequence_gluon(
    # Pointers
    x_ptr, weight_ptr, bias_ptr, lengths_ptr, output_ptr,
    # Shapes
    M: gl.constexpr,
    D_IN: gl.constexpr,
    D_OUT: gl.constexpr,
    # Strides
    x_stride_b: gl.constexpr,
    x_stride_s: gl.constexpr,
    x_stride_d: gl.constexpr,
    weight_stride_in: gl.constexpr,
    weight_stride_out: gl.constexpr,
    bias_stride: gl.constexpr,
    lengths_stride: gl.constexpr,
    output_stride_b: gl.constexpr,
    output_stride_s: gl.constexpr,
    output_stride_d: gl.constexpr,
    # meta-parameters
    TILE_D: gl.constexpr,
    BLOCK_S: gl.constexpr,
    BLOCK_D: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr = 8,
    NUM_WARPS: gl.constexpr = 4,
    EVEN_K: gl.constexpr = True,
    EVEN_D: gl.constexpr = True
):
    # # ------------------- L2 Swizzle Start -------------------
    # # Map program ids `pid` to the block of C it should compute.
    # # This is done in a grouped ordering to promote L2 data reuse.

    # num_pid_m = gl.cdiv(M, BLOCK_S)
    # num_pid_n = gl.cdiv(D_OUT, BLOCK_D)
    # grid_mn = num_pid_m * num_pid_n

    # # Simple direct mapping without swizzle
    # pid = gl.program_id(axis=0)
    # pid = remap_xcd(pid, grid_mn, NUM_XCDS=4)  # 4 XCDs on MI308X
    # pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # # ------------------- L2 Swizzle End -------------------

    # ------------------- No-Swizzle  Start -------------------
    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(D_OUT, BLOCK_D)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    # ------------------- No-Swizzle  End -------------------

    num_pid_m_per_batch = gl.cdiv(M, BLOCK_S)
    pid_b = pid_m // num_pid_m_per_batch
    pid_s = pid_m % num_pid_m_per_batch
    pid_d = pid_n  # 对应 D_OUT

    # Non-Computation part
    real_length = gl.load(lengths_ptr+pid_b * lengths_stride).to(gl.int32)
    row_start = pid_s * BLOCK_S

    if row_start >= real_length or row_start >= M:
        return

    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[16, 4],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[16, 16],
        # instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[1, NUM_WARPS],
    )

    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=2, max_phase=4, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=2, max_phase=4, order=[0, 1]
    )

    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )

    # Prologue: Load first block of A and B
    offs_ak = gl.arange(0, TILE_D, layout=gl.SliceLayout(0, blocked_mk))
    offs_am = (
        pid_m * BLOCK_S
        + gl.arange(0, BLOCK_S, layout=gl.SliceLayout(1, blocked_mk))
    ) % M
    offs_a = pid_b * x_stride_b + \
        offs_am[:, None] * x_stride_s + offs_ak[None, :] * x_stride_d
    if EVEN_K:
        a = gl.amd.cdna3.buffer_load(
            ptr=x_ptr,
            offsets=offs_a,
        )
    else:
        a = gl.amd.cdna3.buffer_load(
            ptr=x_ptr,
            offsets=offs_a,
            mask=(offs_ak[None, :] < D_IN),
        )

    gemm_type = x_ptr.dtype.element_ty
    offs_bk = gl.arange(0, TILE_D, layout=gl.SliceLayout(1, blocked_kn))
    offs_bn = (
        pid_n * BLOCK_D
        + gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, blocked_kn))
    ) % D_OUT
    offs_b = offs_bk[:, None] * weight_stride_in + \
        offs_bn[None, :] * weight_stride_out
    if EVEN_K:
        b = gl.amd.cdna3.buffer_load(
            ptr=weight_ptr,
            offsets=offs_b,
        ).to(gemm_type)
    else:
        b = gl.amd.cdna3.buffer_load(
            ptr=weight_ptr,
            offsets=offs_b,
            mask=(offs_bk[:, None] < D_IN),
        ).to(gemm_type)

    # Create shared memories
    smem_a = gl.allocate_shared_memory(
        x_ptr.type.element_ty, [BLOCK_S, TILE_D], layout=shared_a
    )
    smem_b = gl.allocate_shared_memory(
        weight_ptr.type.element_ty, [TILE_D, BLOCK_D], layout=shared_b
    )

    # LDS write first block of A
    smem_a.store(a)

    acc_dtype = gl.float32
    acc = gl.zeros((BLOCK_S, BLOCK_D),
                   dtype=acc_dtype, layout=mfma_layout)

    # Main Loop: num_stages = 2
    for k in range(0, gl.cdiv(D_IN, TILE_D) - 1):

        # advance pointers for block A and B
        x_ptr += TILE_D * x_stride_d
        weight_ptr += TILE_D * weight_stride_in

        # load next block of A
        if EVEN_K:
            a = gl.amd.cdna3.buffer_load(
                ptr=x_ptr,
                offsets=offs_a,
            )
        else:
            a = gl.amd.cdna3.buffer_load(
                ptr=x_ptr,
                offsets=offs_a,
                mask=(offs_ak[None, :] < D_IN - (k + 1) * TILE_D),
            )

        # LDS write current block of B
        smem_b.store(b)

        # read current block of A from LDS
        cur_a = smem_a.load(layout=dot_a_layout)

        # load next block of B
        if EVEN_K:
            b = gl.amd.cdna3.buffer_load(
                ptr=weight_ptr,
                offsets=offs_b,
            )
        else:
            b = gl.amd.cdna3.buffer_load(
                ptr=weight_ptr,
                offsets=offs_b,
                mask=(offs_bk[:, None] < D_IN - (k + 1) * TILE_D),
            )

        # read current block of B from LDS
        cur_b = smem_b.load(layout=dot_b_layout)

        acc = gl.amd.cdna3.mfma(cur_a, cur_b, acc)

        # write next block of A to LDS
        smem_a.store(a)

    # Epilogue

    # write last block of B to LDS
    smem_b.store(b)

    # read last blocks of A and B from LDS
    cur_a = smem_a.load(layout=dot_a_layout)
    cur_b = smem_b.load(layout=dot_b_layout)

    acc = gl.amd.cdna3.mfma(cur_a, cur_b, acc)

    # add bias
    offsets_bias = (pid_d * BLOCK_D +
                    gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, mfma_layout))) * bias_stride

    bias = gl.amd.cdna3.buffer_load(
        ptr=bias_ptr,
        offsets=offsets_bias,
        mask=offsets_bias < D_OUT,
    )
    acc += bias[None, :]

    c = acc.to(output_ptr.type.element_ty)

    # store block C back to global memory with masks
    offs_cm = pid_m * BLOCK_S + gl.arange(
        0, BLOCK_S, layout=gl.SliceLayout(1, mfma_layout)
    )
    offs_cn = pid_n * BLOCK_D + gl.arange(
        0, BLOCK_D, layout=gl.SliceLayout(0, mfma_layout)
    )
    c_offs = pid_b * output_stride_b + output_stride_s * \
        offs_cm[:, None] + output_stride_d * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < D_OUT)

    gl.amd.cdna3.buffer_store(
        stored_value=c, ptr=output_ptr, offsets=c_offs, mask=c_mask)


def call_addmm_x_BSD_variable_length_tile_d_swizzle_gluon(x, weight, bias, lengths):
    BATCH, MAX_SEQUENCE_LEN, D_IN = x.shape
    D_OUT = weight.shape[0]

    def grid(META):
        num_pid_m = BATCH * triton.cdiv(MAX_SEQUENCE_LEN, META['BLOCK_S'])
        num_pid_n = triton.cdiv(D_OUT, META['BLOCK_D'])
        return (num_pid_m * num_pid_n, )
    TILE_D = 32
    BLOCK_D = 128
    output = torch.empty((BATCH, MAX_SEQUENCE_LEN, D_OUT),
                         device=x.device, dtype=x.dtype)
    torch.library.wrap_triton(_addmm_x_BSD_variable_length_tile_d_swizzle_long_sequence_gluon)[grid](
        x, weight, bias, lengths, output,
        MAX_SEQUENCE_LEN, D_IN, D_OUT,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(1), weight.stride(0),
        bias.stride(0), lengths.stride(0),
        output.stride(0), output.stride(1), output.stride(2),
        EVEN_K=(D_IN % TILE_D == 0),
        EVEN_D=(D_OUT % BLOCK_D == 0)
    )
    return output


@torch.library.triton_op("rtp_lib::variable_length_swizzle_addmm_gluon", mutates_args={})
def rtp_variable_length_swizzle_addmm_gluon(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        lengths: torch.Tensor,
) -> torch.Tensor:
    """
    支持传入序列长度参数的linear(addmm)
    Args:
        x: [B, MAX_SEQUENCE_LEN, D_IN]
        weight: [D_OUT, D_IN]
        bias: [D_OUT]
        lengths: [B, 1] 其中lengths[i][0] 表示第i个batch的序列长度 ∈ [0, MAX_SEQUENCE_LEN]
    Output:
        [B, MAX_SEQUENCE_LEN, D_OUT] - linear计算输出 保证长度范围内fp16计算结果完全正确 长度范围外不保证数值(大概率为0)
    """
    return call_addmm_x_BSD_variable_length_tile_d_swizzle_gluon(x, weight, bias, lengths)


@torch.library.impl("rtp_lib::variable_length_swizzle_addmm_gluon", "cpu")
def rtp_variable_length_swizzle_addmm_cpu(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        lengths: torch.Tensor,
) -> torch.Tensor:
    return F.linear(x, weight, bias)


# nn.Linear如果调用cublas进行计算 本身使用fp16 建议打开fp16优化
torch.library.register_autocast(
    "rtp_lib::variable_length_swizzle_addmm_gluon",
    "cuda",
    torch.float16
)

torch.library.register_autocast(
    "rtp_lib::variable_length_swizzle_addmm_gluon",
    "cuda",
    torch.bfloat16
)


@torch.library.register_fake("rtp_lib::variable_length_swizzle_addmm_gluon")
def rtp_variable_length_swizzle_addmm_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        lengths: torch.Tensor,
) -> torch.Tensor:
    return F.linear(x, weight, bias)


def torch_prof_variable_length_swizzle_addmm_gluon(B, S, DI, DO, REAL_LEN):

    x = (torch.randn(B, S, DI) % 1)
    weight_in = (torch.randn(DO, DI) % 1)
    weight_out = (torch.randn(DI, DO) % 1)
    bias_in = (torch.randn(DO) % 1)
    bias_out = (torch.randn(DI) % 1)
    lengths = torch.Tensor([REAL_LEN] * B).to(torch.int32).unsqueeze(-1)

    x = x.cuda().half()
    weight_in = weight_in.cuda().half()
    weight_out = weight_out.cuda().half()
    bias_in = bias_in.cuda().half()
    bias_out = bias_out.cuda().half()
    lengths = lengths.cuda()

    def fn(x):
        x = call_addmm_x_BSD_variable_length_tile_d_swizzle_gluon(
            x, weight_in, bias_in, lengths)
        x = call_addmm_x_BSD_variable_length_tile_d_swizzle_gluon(
            x, weight_out, bias_out, lengths)
        return x

    for _ in range(10):
        fn(x)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=False, with_stack=True) as p:
        for _ in range(50):
            fn(x)
    torch.cuda.synchronize()
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    p.export_chrome_trace("trace_variable_length_swizzle_addmm_gluon.json")


def validate():
    # x = torch.randn(4, 8, 32) % 1
    # weight = torch.randn(16, 32) % 1
    # bias = torch.randn(16) % 1
    # lengths = torch.Tensor([8, 8, 8, 8]).to(torch.int32)

    # x, weight, bias, lengths = x.cuda(), weight.cuda(), bias.cuda(), lengths.cuda()
    # x, weight, bias = x.half(), weight.half(), bias.half()

    # output = call_addmm_x_BSD_variable_length_tile_d_triton(x, weight, bias, lengths)
    # output_base = F.linear(x, weight, bias)
    # print(torch.max(torch.abs(output - output_base)))
    # print(torch.max(torch.abs(F.layer_norm(output, (4, 8, 16)) - F.layer_norm(output_base, (4, 8, 16)))))

    # import pdb;pdb.set_trace()

    x = torch.randn(4, 16, 3072) % 1
    weight = torch.randn(768, 3072) % 1
    bias = torch.randn(768) % 1
    lengths = torch.Tensor([20, 10, 20, 20]).to(torch.int32).unsqueeze(-1)

    x, weight, bias, lengths = x.cuda(), weight.cuda(), bias.cuda(), lengths.cuda()
    x, weight, bias = x.half(), weight.half(), bias.half()

    res = call_addmm_x_BSD_variable_length_tile_d_swizzle_gluon(
        x, weight, bias, lengths)
    res_base = F.linear(x, weight, bias)

    torch.allclose(res, res_base, atol=1e-3, rtol=1e-3)
    # for i in range(16):print(torch.max(torch.abs(res[0][i] - res_base[0][i])), i)

    # import pdb
    # pdb.set_trace()


if __name__ == "__main__":

    # validate()
    # torch_prof_variable_length_swizzle_addmm(80, 248, 3072, 768, 100)
    torch_prof_variable_length_swizzle_addmm_gluon(80, 248, 3072, 768, 80)
