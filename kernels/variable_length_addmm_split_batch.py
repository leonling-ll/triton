import torch
import triton
import triton.language as tl

import numpy as np
import math
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity


if torch.version.hip:
    autotune_configs = [
        # triton.Config({"TILE_D": 32, "BLOCK_B": 8, "BLOCK_S": 32, "BLOCK_D": 128, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 2}, num_warps=8, num_stages=3),
        # triton.Config({"TILE_D": 32, "BLOCK_B": 8, "BLOCK_S": 32, "BLOCK_D": 128, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 2}, num_warps=4, num_stages=3),
        # triton.Config({"TILE_D": 32, "BLOCK_B": 8, "BLOCK_S": 32, "BLOCK_D": 128, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 2}, num_warps=8, num_stages=2),

        # triton.Config({"TILE_D": 32, "BLOCK_B": 8, "BLOCK_S": 32, "BLOCK_D": 128, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 2}, num_warps=8, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_B": 8, "BLOCK_S": 32, "BLOCK_D": 128, 'matrix_instr_nonkdim': 16, "waves_per_eu": 1, "kpack": 1}, num_warps=4, num_stages=1),
        # triton.Config({"TILE_D": 32, "BLOCK_B": 8, "BLOCK_S": 32, "BLOCK_D": 128, 'matrix_instr_nonkdim': 16, "waves_per_eu": 0, "kpack": 1}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_B": 8, "BLOCK_S": 32, "BLOCK_D": 128, 'matrix_instr_nonkdim': 16, "waves_per_eu": 0, "kpack": 1}, num_warps=8, num_stages=2),
        
        # triton.Config({"TILE_D": 32, "BLOCK_B": 8, "BLOCK_S": 32, "BLOCK_D": 128, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 2}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_B": 8, "BLOCK_S": 32, "BLOCK_D": 128, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 2}, num_warps=2, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_B": 8, "BLOCK_S": 64, "BLOCK_D": 128, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 2}, num_warps=4, num_stages=2),


        # triton.Config({"TILE_D": 32, "BLOCK_B": 4, "BLOCK_S": 32, "BLOCK_D": 128, 'matrix_instr_nonkdim': 16, "waves_per_eu": 0, "kpack": 2}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_B": 4, "BLOCK_S": 32, "BLOCK_D": 128, 'matrix_instr_nonkdim': 16, "waves_per_eu": 2, "kpack": 2}, num_warps=4, num_stages=2),
        triton.Config({"TILE_D": 32, "BLOCK_B": 4, "BLOCK_S": 32, "BLOCK_D": 128, 'matrix_instr_nonkdim': 16, "waves_per_eu": 4, "kpack": 2}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_B": 4, "BLOCK_S": 64, "BLOCK_D": 128, 'matrix_instr_nonkdim': 16, "waves_per_eu": 0, "kpack": 2}, num_warps=4, num_stages=2),
        # triton.Config({"TILE_D": 32, "BLOCK_B": 8, "BLOCK_S": 32, "BLOCK_D": 128, 'matrix_instr_nonkdim': 16, "waves_per_eu": 0, "kpack": 2}, num_warps=4, num_stages=2),
    ]
else:
    autotune_configs = [
        triton.Config({"TILE_D": 32, "BLOCK_B": 8, "BLOCK_S": 32, "BLOCK_D": 128}, num_warps=8, num_stages=3),
        triton.Config({"TILE_D": 32, "BLOCK_B": 8, "BLOCK_S": 32, "BLOCK_D": 128}, num_warps=8, num_stages=3),
    ]



@triton.autotune(
    configs=autotune_configs,
    key=["M", "D_IN", "D_OUT"],
)
@triton.jit
def addmm_x_BSD_variable_length_tile_d_split_batch_triton(
    x_ptr, weight_ptr, bias_ptr, lengths_ptr, output_ptr,
    M: tl.constexpr, D_IN: tl.constexpr, D_OUT: tl.constexpr, 
    x_stride_b: tl.constexpr, x_stride_s: tl.constexpr, x_stride_d: tl.constexpr,
    weight_stride_in: tl.constexpr, weight_stride_out: tl.constexpr, 
    bias_stride: tl.constexpr, lengths_stride:tl.constexpr, 
    output_stride_b: tl.constexpr, output_stride_s: tl.constexpr, output_stride_d: tl.constexpr,
    BATCH, TILE_D: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr, 
):
# -------------- 三维切分 不适合swizzle ---------------
    gemm_type = x_ptr.dtype.element_ty

    pid_b, pid_s, pid_d = tl.program_id(2), tl.program_id(1), tl.program_id(0)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < BATCH

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < M

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_OUT

    real_lengths = tl.load(
            lengths_ptr + offs_b * lengths_stride, 
            mask = offs_b < BATCH,
            other = 0
    ).to(tl.int32)

    output_ptrs = output_ptr + \
        offs_b[:, None, None] * output_stride_b + \
        offs_s[None, :, None] * output_stride_s + \
        offs_d[None, None, :] * output_stride_d

    max_real_length = tl.max(real_lengths, axis=0) 
    row_start = pid_s * BLOCK_S
    if row_start >= max_real_length or row_start >= M:
        tl.store(
            output_ptrs, 
            tl.zeros((BLOCK_B, BLOCK_S, BLOCK_D), 
            dtype=output_ptr.dtype.element_ty), 
            mask = mask_b[:, None, None] & mask_s[None, :, None] & mask_d[None, None, :],
            # cache_modifier=".wt"
        )
        return 
    
    acc = tl.zeros((BLOCK_B * BLOCK_S, BLOCK_D), dtype=tl.float32)

    # 在d_in维度上进行tiling
    # for start_d_in in range(0, D_IN, TILE_D):
    for start_d_in in tl.range(0, D_IN, TILE_D, num_stages=2):
        offs_d_in = start_d_in + tl.arange(0, TILE_D)
        mask_d_in = offs_d_in < D_IN

        x = tl.load(
            x_ptr + offs_b[:, None, None] * x_stride_b + offs_s[None, :, None] * x_stride_s + offs_d_in[None, None, :] * x_stride_d, 
            mask = mask_b[:, None, None] & mask_s[None, :, None] & mask_d_in[None, None, :], 
            other = 0.0
        )
        weight = tl.load(
            weight_ptr +  offs_d_in[:, None] * weight_stride_in + offs_d[None, :] * weight_stride_out, 
            mask = mask_d_in[:, None] & mask_d[None, :], 
            other = 0.0
        ).to(gemm_type)

        x_2d = tl.reshape(x, (BLOCK_B * BLOCK_S, TILE_D))
        acc = tl.dot(x_2d, weight, acc=acc)
        # acc += x_dot
    acc = tl.reshape(acc, (BLOCK_B, BLOCK_S, BLOCK_D))
    bias = tl.load(
        bias_ptr + (pid_d *  BLOCK_D + tl.arange(0, BLOCK_D)) * bias_stride, 
        mask = (pid_d *  BLOCK_D + tl.arange(0, BLOCK_D)) < D_OUT, 
        other = 0.0
    )
    acc += bias[None, None, :]

    # write back
    tl.store(
        output_ptrs, 
        acc.to(output_ptr.dtype.element_ty), 
        mask = mask_b[:, None, None] & mask_s[None, :, None] & mask_d[None, None, :],
        # cache_modifier=".wt"
    )


def call_addmm_x_BSD_variable_length_tile_d_split_batch_triton(x, weight, bias, lengths):
    BATCH, MAX_SEQUENCE_LEN, D_IN = x.shape
    D_OUT = weight.shape[0]

    def grid(META):
        return (
            triton.cdiv(D_OUT, META['BLOCK_D']),
            triton.cdiv(MAX_SEQUENCE_LEN, META['BLOCK_S']),
            triton.cdiv(BATCH, META["BLOCK_B"]),
        )   
    output = torch.empty((BATCH, MAX_SEQUENCE_LEN, D_OUT), device=x.device, dtype = x.dtype)

    torch.library.wrap_triton(addmm_x_BSD_variable_length_tile_d_split_batch_triton)[grid](
        x, weight, bias, lengths, output,
        MAX_SEQUENCE_LEN, D_IN, D_OUT, 
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(1), weight.stride(0),
        bias.stride(0), lengths.stride(0),
        output.stride(0), output.stride(1), output.stride(2),
        BATCH
    )
    return output

@torch.library.triton_op("rtp_lib::variable_length_split_batch_addmm", mutates_args={})
def rtp_variable_length_split_batch_addmm(
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
    return call_addmm_x_BSD_variable_length_tile_d_split_batch_triton(x, weight, bias, lengths)

@torch.library.impl("rtp_lib::variable_length_split_batch_addmm", "cpu")
def rtp_variable_length_split_batch_addmm_cpu(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        lengths: torch.Tensor,
) -> torch.Tensor:
    return F.linear(x, weight, bias)

# nn.Linear如果调用cublas进行计算 本身使用fp16 建议打开fp16优化
torch.library.register_autocast(
    "rtp_lib::variable_length_split_batch_addmm",
    "cuda",
    torch.float16
)

torch.library.register_autocast(
    "rtp_lib::variable_length_split_batch_addmm",
    "cuda",
    torch.bfloat16
)

@torch.library.register_fake("rtp_lib::variable_length_split_batch_addmm")
def rtp_variable_length_split_batch_addmm_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        lengths: torch.Tensor,
) -> torch.Tensor:
    return F.linear(x, weight, bias)


def torch_prof_variable_length_split_batch(B, S, DI, DO, REAL_LEN):

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
        x = call_addmm_x_BSD_variable_length_tile_d_split_batch_triton(x, weight_in, bias_in, lengths)
        x = call_addmm_x_BSD_variable_length_tile_d_split_batch_triton(x, weight_out, bias_out, lengths)

    # warmup
    for _ in range(10):
        fn(x)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=False, with_stack=True) as p:
        for _ in range(50):
            fn(x)
    torch.cuda.synchronize()
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    p.export_chrome_trace("trace_variable_length_split_batch.json")


bench_configs = []
bench_configs.append(triton.testing.Benchmark(
    x_names=["B", "S", "DI", "DO", "REAL_LEN"],
    x_vals=[
        [80, 100, 3072, 768, 100],   #
        [80, 100, 3072, 768, 50],   #
        [80, 100, 768, 3072, 100],   #
        [80, 100, 768, 3072, 50],   #
    ],
    line_arg="provider",
    line_vals=["triton"],
    line_names=["Triton"],
    styles=[("green", "-"), ("blue", "-")],
    plot_name="triton_copy_addmm_trans_performance",
    args={}
))


@triton.testing.perf_report(bench_configs)
def triton_benchmark(B, S, DI, DO, REAL_LEN, provider):
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

    # x = torch.randn(4, 25, 3072) % 1
    # weight = torch.randn(768, 3072) % 1
    # bias = torch.randn(768) % 1
    # lengths = torch.Tensor([15, 10, 20, 20]).to(torch.int32).unsqueeze(-1)

    # x, weight, bias, lengths = x.cuda(), weight.cuda(), bias.cuda(), lengths.cuda()
    # x, weight, bias = x.half(), weight.half(), bias.half()

    x = (torch.randn(B, S, DI) % 1)
    weight = (torch.randn(DO, DI) % 1)
    bias = (torch.randn(DO) % 1)
    lengths = torch.Tensor([REAL_LEN] * B).to(torch.int32).unsqueeze(-1)

    x = x.cuda(0).half()
    weight = weight.cuda().half()
    bias = bias.cuda().half()
    lengths = lengths.cuda()

    quantiles = [0.5, 0.1, 0.9]

    fn = lambda: call_addmm_x_BSD_variable_length_tile_d_split_batch_triton(x, weight, bias, lengths)
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    def tim(ms): return ms
    return tim(ms), tim(max_ms), tim(min_ms)


if __name__ == "__main__":
    # triton_benchmark.run(show_plots=False, print_data=True)
    torch_prof_variable_length_split_batch(80, 248, 3072, 768, 80)
    # torch_prof_variable_length_split_batch(80, 100, 3072, 768, 50)
