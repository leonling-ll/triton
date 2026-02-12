import torch
import triton
import triton.language as tl

import numpy as np
import math
import torch.nn.functional as F

from torch.profiler import profile, record_function, ProfilerActivity

# BLOCK_S比较大 效率非常高 在序列长度大于100 BatchSize=80 [3072 768]线性投影中
# 即便真实序列长度等于最大序列长度 与原生实现速度几乎持平
# 适用于序列长度比较大 >= 512的场景
# 如果序列最大长度在[256 ~ 512]之间 BLOCK_S可以选取64

# !!!注意 因为需要在 length 维度上按照真实长度进行节约计算 BLOCK_S不能太大 下面按照业务需求指定为64
if torch.version.hip:
    autotune_configs = [
        # triton.Config({"TILE_D": 32, "BLOCK_S": 64, "BLOCK_D":128, "GROUP_SIZE_M": 16, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 2}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_S": 64, "BLOCK_D":128, "GROUP_SIZE_M": 16, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 4}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_S": 64, "BLOCK_D":128, "GROUP_SIZE_M": 16, 'matrix_instr_nonkdim': 32, "waves_per_eu": 4, "kpack": 4}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 64, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 64, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 2}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 64, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 4}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 64, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8, 'matrix_instr_nonkdim': 32, "waves_per_eu": 4, "kpack": 4}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 64, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 2}, num_warps=2, num_stages=2),
        # triton.Config({"TILE_D": 64, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 4}, num_warps=2, num_stages=2),
        # triton.Config({"TILE_D": 64, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8, 'matrix_instr_nonkdim': 32, "waves_per_eu": 4, "kpack": 4}, num_warps=2, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8,
        #               'matrix_instr_nonkdim': 16, "waves_per_eu": 0, "kpack": 2}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 64, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8,
        #               'matrix_instr_nonkdim': 16, "waves_per_eu": 2, "kpack": 2}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8,
        #               'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 1}, num_warps=4, num_stages=2),
        triton.Config({"TILE_D": 32, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8,
                      'matrix_instr_nonkdim': 16, "waves_per_eu": 0, "kpack": 2}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 4}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8, 'matrix_instr_nonkdim': 32, "waves_per_eu": 4, "kpack": 4}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 2}, num_warps=2, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 4}, num_warps=2, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8, 'matrix_instr_nonkdim': 32, "waves_per_eu": 4, "kpack": 4}, num_warps=2, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_S": 128, "BLOCK_D":128, "GROUP_SIZE_M": 16, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 2}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_S": 128, "BLOCK_D":128, "GROUP_SIZE_M": 16, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 4}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_S": 128, "BLOCK_D":128, "GROUP_SIZE_M": 16, 'matrix_instr_nonkdim': 32, "waves_per_eu": 4, "kpack": 4}, num_warps=4, num_stages=2),

        # triton.Config({"TILE_D": 32, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8, 'matrix_instr_nonkdim': 16, "waves_per_eu": 0, "kpack": 1}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 8, 'matrix_instr_nonkdim': 16, "waves_per_eu": 0, "kpack": 1}, num_warps=8, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 16, 'matrix_instr_nonkdim': 16, "waves_per_eu": 0, "kpack": 1}, num_warps=8, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 16, 'matrix_instr_nonkdim': 16, "waves_per_eu": 0, "kpack": 1}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_S": 64, "BLOCK_D": 128, "GROUP_SIZE_M": 4, 'matrix_instr_nonkdim': 16, "waves_per_eu": 0, "kpack": 1}, num_warps=8, num_stages=2),
    ]
else:
    autotune_configs = [
        # triton.Config({"TILE_D": 64, "BLOCK_S": 128, "BLOCK_D": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"TILE_D": 64, "BLOCK_S": 64, "BLOCK_D": 128,
                      "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
    ]

# TILE_D: K, BLOCK_S:M, BLOCK_D:N


@triton.autotune(
    configs=autotune_configs,
    key=["M", "D_IN", "D_OUT"],
)
@triton.jit
def _addmm_x_BSD_variable_length_tile_d_swizzle_long_sequence_triton(
    x_ptr, weight_ptr, bias_ptr, lengths_ptr, output_ptr,
    M: tl.constexpr, D_IN: tl.constexpr, D_OUT: tl.constexpr,
    x_stride_b: tl.constexpr, x_stride_s: tl.constexpr, x_stride_d: tl.constexpr,
    weight_stride_in: tl.constexpr, weight_stride_out: tl.constexpr,
    bias_stride: tl.constexpr, lengths_stride: tl.constexpr,
    output_stride_b: tl.constexpr, output_stride_s: tl.constexpr, output_stride_d: tl.constexpr,
    TILE_D: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 8,
    EVEN_K: tl.constexpr = True,
    EVEN_D: tl.constexpr = True
):
    gemm_type = x_ptr.dtype.element_ty
    num_pid_n = tl.cdiv(D_OUT, BLOCK_D)
    total_programs = tl.num_programs(0)
    num_pid_m = total_programs // num_pid_n

    # # -------- L2 Cache Swizzling 加速 ----------
    # pid = tl.program_id(0)

    # num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # group_id = pid // num_pid_in_group
    # first_pid_m = group_id * GROUP_SIZE_M
    # group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    # pid_m = first_pid_m + (pid % group_size_m)
    # pid_n = (pid % num_pid_in_group) // group_size_m

    # ------------- Swizzle 结束-----------------

    # Simple direct mapping without swizzle
    pid = tl.program_id(0)
    total_programs = tl.num_programs(0)
    num_pid_m = total_programs // num_pid_n
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # ------------- No-Swizzle 结束-----------------

    num_pid_m_per_batch = tl.cdiv(M, BLOCK_S)
    pid_b = pid_m // num_pid_m_per_batch
    pid_s = pid_m % num_pid_m_per_batch
    pid_d = pid_n  # 对应 D_OUT

    real_length = tl.load(lengths_ptr + pid_b * lengths_stride).to(tl.int32)
    row_start = pid_s * BLOCK_S
    # if row_start >= real_length or row_start >= M:
    #     return
    if row_start >= real_length or row_start >= M:
        # 直接写0到输出
        out_block_ptr = tl.make_block_ptr(
            base=output_ptr + pid_b * output_stride_b,
            shape=(M, D_OUT),
            strides=(output_stride_s, output_stride_d),
            offsets=(pid_s * BLOCK_S, pid_d * BLOCK_D),
            block_shape=(BLOCK_S, BLOCK_D),
            order=(1, 0)
        )
        zeros = tl.zeros((BLOCK_S, BLOCK_D), dtype=output_ptr.dtype.element_ty)
        if EVEN_D:
            tl.store(out_block_ptr, zeros, boundary_check=(0,))
        else:
            tl.store(out_block_ptr, zeros, boundary_check=(0, 1))
        return

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr + pid_b * x_stride_b,
        shape=(M, D_IN),
        strides=(x_stride_s, x_stride_d),
        offsets=(pid_s * BLOCK_S, 0),
        block_shape=(BLOCK_S, TILE_D),
        order=(1, 0)
    )
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D_IN, D_OUT),
        strides=(weight_stride_in, weight_stride_out),
        offsets=(0, pid_d * BLOCK_D),
        block_shape=(TILE_D, BLOCK_D),
        order=(0, 1)
    )

    acc = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)

    # 在d_in维度上进行tiling
    for start_d_in in range(0, D_IN, TILE_D):
    # for start_d_in in tl.range(0, D_IN, TILE_D, num_stages=2):
        if EVEN_K:
            x = tl.load(x_block_ptr, boundary_check=(0,))
            weight = tl.load(weight_block_ptr,
                             boundary_check=(1,)).to(gemm_type)
        else:
            x = tl.load(x_block_ptr, boundary_check=(0, 1))
            weight = tl.load(weight_block_ptr,
                             boundary_check=(0, 1)).to(gemm_type)

        acc = tl.dot(x, weight, acc)

        x_block_ptr = tl.advance(x_block_ptr, (0, TILE_D))
        weight_block_ptr = tl.advance(weight_block_ptr, (TILE_D, 0))

    if EVEN_D:
        bias_mask = None
        bias = tl.load(bias_ptr + pid_d * BLOCK_D * bias_stride)
    else:
        bias_mask = pid_d * BLOCK_D + tl.arange(0, BLOCK_D) < D_OUT
        bias = tl.load(
            bias_ptr + (pid_d * BLOCK_D + tl.arange(0, BLOCK_D)) * bias_stride,
            mask=bias_mask,
            other=0.0
        )
    acc += bias[None, :]

    # write back
    out_block_ptr = tl.make_block_ptr(
        base=output_ptr + pid_b * output_stride_b,
        shape=(M, D_OUT),
        strides=(output_stride_s, output_stride_d),
        offsets=(pid_s * BLOCK_S, pid_d * BLOCK_D),
        block_shape=(BLOCK_S, BLOCK_D),
        order=(1, 0)
    )
    if EVEN_D:
        tl.store(out_block_ptr, acc.to(
            out_block_ptr.dtype.element_ty), boundary_check=(0,))
    else:
        tl.store(out_block_ptr, acc.to(
            out_block_ptr.dtype.element_ty), boundary_check=(0, 1))


def call_addmm_x_BSD_variable_length_tile_d_swizzle_triton(x, weight, bias, lengths):
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
    torch.library.wrap_triton(_addmm_x_BSD_variable_length_tile_d_swizzle_long_sequence_triton)[grid](
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


@torch.library.triton_op("rtp_lib::variable_length_swizzle_addmm", mutates_args={})
def rtp_variable_length_swizzle_addmm(
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
    return call_addmm_x_BSD_variable_length_tile_d_swizzle_triton(x, weight, bias, lengths)


@torch.library.impl("rtp_lib::variable_length_swizzle_addmm", "cpu")
def rtp_variable_length_swizzle_addmm_cpu(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        lengths: torch.Tensor,
) -> torch.Tensor:
    return F.linear(x, weight, bias)


# nn.Linear如果调用cublas进行计算 本身使用fp16 建议打开fp16优化
torch.library.register_autocast(
    "rtp_lib::variable_length_swizzle_addmm",
    "cuda",
    torch.float16
)

torch.library.register_autocast(
    "rtp_lib::variable_length_swizzle_addmm",
    "cuda",
    torch.bfloat16
)


@torch.library.register_fake("rtp_lib::variable_length_swizzle_addmm")
def rtp_variable_length_swizzle_addmm_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        lengths: torch.Tensor,
) -> torch.Tensor:
    return F.linear(x, weight, bias)


def torch_prof_variable_length_swizzle_addmm(B, S, DI, DO, REAL_LEN):

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
        x = call_addmm_x_BSD_variable_length_tile_d_swizzle_triton(x, weight_in, bias_in, lengths)
        x = call_addmm_x_BSD_variable_length_tile_d_swizzle_triton(x, weight_out, bias_out, lengths)
        return x
        
    for _ in range(10):
        fn(x)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=False, with_stack=True) as p:
        for _ in range(50):
            fn(x)
    torch.cuda.synchronize()
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    p.export_chrome_trace("trace_variable_length_swizzle_addmm.json")


def old_main():
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

    real_length = 100
    x = torch.randn(4, 16, 3072) % 1
    x = torch.randn(80, 100, 3072) % 1
    weight = torch.randn(768, 3072) % 1
    bias = torch.randn(768) % 1
    lengths = torch.Tensor([real_length]*80).to(torch.int32).unsqueeze(-1)

    x, weight, bias, lengths = x.cuda(), weight.cuda(), bias.cuda(), lengths.cuda()
    x, weight, bias = x.half(), weight.half(), bias.half()

    res = call_addmm_x_BSD_variable_length_tile_d_swizzle_triton(
        x, weight, bias, lengths)
    res_base = F.linear(x, weight, bias)

    for i in range(16):
        print(torch.max(torch.abs(res[0][i] - res_base[0][i])), i)

    import pdb
    pdb.set_trace()


bench_configs = []
bench_configs.append(triton.testing.Benchmark(
    x_names=["B", "S", "DI", "DO", "REAL_LEN"],
    x_vals=[
        [80, 100, 3072, 768, 100],   #
        [80, 100, 3072, 768, 50],   #
    ],
    line_arg="provider",
    line_vals=["triton", "torch"],
    line_names=["Triton", "Torch"],
    styles=[("green", "-"), ("blue", "-")],
    plot_name="triton_copy_addmm_trans_performance",
    args={}
))


@triton.testing.perf_report(bench_configs)
def triton_benchmark(B, S, DI, DO, REAL_LEN, provider):

    x = (torch.randn(B, S, DI) % 1)
    weight = (torch.randn(DO, DI) % 1)
    bias = (torch.randn(DO) % 1)
    lengths = torch.Tensor([REAL_LEN] * B).to(torch.int32).unsqueeze(-1)

    x = x.cuda().half()
    weight = weight.cuda().half()
    bias = bias.cuda().half()
    lengths = lengths.cuda()

    quantiles = [0.5, 0.1, 0.9]

    def triton_fn(): return call_addmm_x_BSD_variable_length_tile_d_swizzle_triton(
        x, weight, bias, lengths)

    def torch_fn(): return F.linear(x, weight, bias)

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(triton_fn, quantiles=quantiles)
    elif provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(torch_fn, quantiles=quantiles)
    else:
        raise ValueError(f"Invalid provider: {provider}")

    def tim(ms): return ms
    return tim(ms), tim(max_ms), tim(min_ms)


if __name__ == "__main__":

    # old_main()
    # triton_benchmark.run(show_plots=False, print_data=True)
    # torch_prof_variable_length_swizzle_addmm(80, 248, 3072, 768, 100)
    torch_prof_variable_length_swizzle_addmm(80, 248, 3072, 768, 80)
