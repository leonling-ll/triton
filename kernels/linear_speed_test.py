
from variable_length_addmm_swizzle import call_addmm_x_BSD_variable_length_tile_d_swizzle_triton
from variable_length_addmm_split_batch import call_addmm_x_BSD_variable_length_tile_d_split_batch_triton
from variable_length_addmm_swizzle_gluon import call_addmm_x_BSD_variable_length_tile_d_swizzle_gluon

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity


class TestModel(nn.Module):
    def __init__(self, d_in, d_out, use_cublas=True, use_swizzle=False, use_gluon=False):
        super().__init__()
        self.weight_in = nn.Parameter(torch.randn(d_out, d_in) % 1)
        self.weight_out = nn.Parameter(torch.randn(d_in, d_out) % 1)
        self.bias_in = nn.Parameter(torch.randn(d_out) % 1)
        self.bias_out = nn.Parameter(torch.randn(d_in) % 1)

        self.weight_in.requires_grad = False
        self.weight_out.requires_grad = False
        self.bias_in.requires_grad = False
        self.bias_out.requires_grad = False

        self.use_cublas = use_cublas
        self.use_swizzle = use_swizzle
        self.use_gluon = use_gluon

    def forward(self, x, lengths):
        for i in range(10):
            if self.use_cublas:
                x = F.linear(x, self.weight_in, self.bias_in)
                x = F.layer_norm(x, (x.shape[-1],))
                x = F.linear(x, self.weight_out, self.bias_out)
                x = F.layer_norm(x, (x.shape[-1],))
            elif self.use_swizzle and not self.use_gluon:
                x = torch.ops.rtp_lib.variable_length_swizzle_addmm(
                    x, self.weight_in, self.bias_in, lengths)
                x = F.layer_norm(x, (x.shape[-1],))
                x = torch.ops.rtp_lib.variable_length_swizzle_addmm(
                    x, self.weight_out, self.bias_out, lengths)
                x = F.layer_norm(x, (x.shape[-1],))
            elif self.use_swizzle and self.use_gluon:
                x = call_addmm_x_BSD_variable_length_tile_d_swizzle_gluon(
                    x, self.weight_in, self.bias_in, lengths)
                x = F.layer_norm(x, (x.shape[-1],))
                x = call_addmm_x_BSD_variable_length_tile_d_swizzle_gluon(
                    x, self.weight_out, self.bias_out, lengths)
                x = F.layer_norm(x, (x.shape[-1],))
            else:
                x = torch.ops.rtp_lib.variable_length_split_batch_addmm(
                    x, self.weight_in, self.bias_in, lengths)
                x = F.layer_norm(x, (x.shape[-1],))
                x = torch.ops.rtp_lib.variable_length_split_batch_addmm(
                    x, self.weight_out, self.bias_out, lengths)
                x = F.layer_norm(x, (x.shape[-1],))

        return x


def main(seq_len, real_length):
    model_linear = TestModel(768, 3072, True).to("cuda").half()
    model_swizzle = TestModel(768, 3072, False, True).to("cuda").half()
    model_swizzle_gluon = TestModel(768, 3072, False, True, True).to("cuda").half()
    model_split_batch = TestModel(768, 3072, False, False).to("cuda").half()

    # real_length = 50
    # real_length = 100

    x = torch.randn(80, seq_len, 768) % 1
    lengths = torch.Tensor([real_length]*80).to(torch.int32).unsqueeze(-1)

    # 模拟一下复杂的length情况
    # lengths = torch.randint(20, 30, (80, 1)).to(torch.int32).unsqueeze(-1)
    # lengths[0][0] = 50
    # lengths[0][-1] = 62

    store_folder = "profilers_" + str(int(lengths[0][0].item()))
    os.makedirs(store_folder, exist_ok=True)

    x, lengths = x.to("cuda"), lengths.to("cuda")
    x = x.half()

    # # ----------------- Model Linear -----------------
    # with torch.inference_mode():
    #     exported_model = torch.export.export(
    #         model_linear,
    #         args=(),
    #         kwargs={"x": x, "lengths": lengths},
    #         dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC},
    #                         "lengths": {0: torch.export.Dim.DYNAMIC}},
    #     )

    #     aoti_linear_path = torch._inductor.aoti_compile_and_package(
    #         exported_model,
    #         package_path=os.path.join("./aoti", "linear.pt2"),
    #         inductor_configs={"max_autotune": True},
    #     )

    # aoti_model = torch._inductor.aoti_load_package(
    #     aoti_linear_path, device_index=0)
    # for _ in range(5):
    #     res1 = aoti_model(**{"x": x, "lengths": lengths})

    # times = []
    # # 循环100次测试耗时
    # for _ in range(100):
    #     start = torch.cuda.Event(enable_timing=True)
    #     end = torch.cuda.Event(enable_timing=True)
    #     start.record()
    #     res1 = aoti_model(**{"x": x, "lengths": lengths})
    #     end.record()
    #     torch.cuda.synchronize()
    #     times.append(start.elapsed_time(end))
    # print(f"standard linear cost time {sum(times) / len(times)}")

    # with torch.profiler.profile() as p:
    #     for _ in range(5):
    #         res1 = aoti_model(**{"x": x, "lengths": lengths})
    # torch.cuda.synchronize()

    # print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # p.export_chrome_trace(path=os.path.join(store_folder, "linear.json"))

    # # ----------------- Model Split Batch -----------------
    # with torch.inference_mode():
    #     exported_model = torch.export.export(
    #         model_split_batch,
    #         args=(),
    #         kwargs=dict(x=x, lengths=lengths),
    #         dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC},
    #                         "lengths": {0: torch.export.Dim.DYNAMIC}},
    #     )

    #     aoti_linear_path = torch._inductor.aoti_compile_and_package(
    #         exported_model,
    #         package_path=os.path.join("./aoti", "linear.pt2"),
    #         inductor_configs={"max_autotune": True},
    #     )

    # # aoti_linear_path = "./aoti/linear.pt2"
    # aoti_model = torch._inductor.aoti_load_package(
    #     aoti_linear_path, device_index=0)
    # for _ in range(5):
    #     res1 = aoti_model(**{"x": x, "lengths": lengths})
    # times = []
    # # 循环100次测试耗时
    # for _ in range(100):
    #     start = torch.cuda.Event(enable_timing=True)
    #     end = torch.cuda.Event(enable_timing=True)
    #     start.record()
    #     res1 = aoti_model(**{"x": x, "lengths": lengths})
    #     end.record()
    #     torch.cuda.synchronize()
    #     times.append(start.elapsed_time(end))
    # print(f"split_batch cost time {sum(times) / len(times)}")

    # with torch.profiler.profile() as p:
    #     for _ in range(5):
    #         res1 = aoti_model(**{"x": x, "lengths": lengths})
    # torch.cuda.synchronize()

    # print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # p.export_chrome_trace(path=os.path.join(store_folder, "split_batch.json"))

    # # ----------------- Model Swizzle -----------------
    # with torch.inference_mode():
    #     exported_model = torch.export.export(
    #         model_swizzle,
    #         args=(),
    #         kwargs=dict(x=x, lengths=lengths),
    #         dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC},
    #                         "lengths": {0: torch.export.Dim.DYNAMIC}},
    #     )

    #     aoti_linear_path = torch._inductor.aoti_compile_and_package(
    #         exported_model,
    #         package_path=os.path.join("./aoti", "linear.pt2"),
    #         inductor_configs={"max_autotune": True},
    #     )

    # aoti_model = torch._inductor.aoti_load_package(
    #     aoti_linear_path, device_index=0)
    # for _ in range(5):
    #     res1 = aoti_model(**{"x": x, "lengths": lengths})
    # times = []
    # # 循环100次测试耗时
    # for _ in range(100):
    #     start = torch.cuda.Event(enable_timing=True)
    #     end = torch.cuda.Event(enable_timing=True)
    #     start.record()
    #     res1 = aoti_model(**{"x": x, "lengths": lengths})
    #     end.record()
    #     torch.cuda.synchronize()
    #     times.append(start.elapsed_time(end))
    # print(f"swizzle cost time {sum(times) / len(times)}")

    # with torch.profiler.profile() as p:
    #     for _ in range(5):
    #         res1 = aoti_model(**{"x": x, "lengths": lengths})
    # torch.cuda.synchronize()

    # print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # p.export_chrome_trace(path=os.path.join(store_folder, "swizzle.json"))

    # ----------------- Model Swizzle Gluon -----------------
    with torch.inference_mode():
        exported_model = torch.export.export(
            model_swizzle_gluon,
            args=(),
            kwargs=dict(x=x, lengths=lengths),
            dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC},
                            "lengths": {0: torch.export.Dim.DYNAMIC}},
        )

        aoti_linear_path = torch._inductor.aoti_compile_and_package(
            exported_model,
            package_path=os.path.join("./aoti", "linear.pt2"),
            inductor_configs={"max_autotune": True},
        )

    aoti_model = torch._inductor.aoti_load_package(
        aoti_linear_path, device_index=0)
    for _ in range(5):
        res1 = aoti_model(**{"x": x, "lengths": lengths})
    times = []
    # 循环100次测试耗时
    for _ in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        res1 = aoti_model(**{"x": x, "lengths": lengths})
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    print(f"swizzle gluon cost time {sum(times) / len(times)}")

    with torch.profiler.profile() as p:
        for _ in range(5):
            res1 = aoti_model(**{"x": x, "lengths": lengths})
    torch.cuda.synchronize()

    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    p.export_chrome_trace(path=os.path.join(store_folder, "swizzle_gluon.json"))



if __name__ == "__main__":
    main(248, 80)