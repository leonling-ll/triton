"""Benchmark public AITER MXFP4 wrappers with the local Triton checkout."""

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import triton
from aiter.ops.shuffle import shuffle_scale, shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
    gemm_afp4wfp4,
    gemm_afp4wfp4_preshuffled_scales,
    gemm_afp4wfp4_preshuffle,
)
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
    generate_gemm_afp4wfp4_inputs,
    run_torch,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", nargs=3, type=int, default=[51200, 106496, 16384])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--include-gluon", action="store_true")
    parser.add_argument("--accumulation-form", choices=["explicit", "plus_equal"], default="explicit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert Path(triton.__file__).resolve().is_relative_to("/home/leling/triton"), triton.__file__
    M, N, K = args.shape
    if args.accumulation_form == "plus_equal":
        kernels = importlib.import_module("aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4")
        # Change only the in-memory kernel source; the AITER checkout stays intact.
        for attr in ("_gemm_afp4wfp4_kernel", "_gemm_afp4wfp4_kernel_preshuffle_scales",
                     "_gemm_afp4wfp4_preshuffle_kernel"):
            fn = getattr(kernels, attr)
            while not isinstance(fn, triton.runtime.JITFunction):
                fn = fn.fn
            original = 'accumulator = tl.dot_scaled(\n                a, a_scales, "e2m1", b, b_scales, "e2m1", accumulator\n            )'
            replacement = 'accumulator += tl.dot_scaled(\n                a, a_scales, "e2m1", b, b_scales, "e2m1"\n            )'
            assert fn.src.count(original) == 1, attr
            fn._unsafe_update_src(fn.src.replace(original, replacement))
    result = {
        "shape": args.shape,
        "triton": triton.__version__,
        "triton_path": triton.__file__,
        "torch": torch.__version__,
        "device": str(torch.cuda.get_device_properties(0)),
        "old_preshuffle_env": os.environ.get("TRITON_HIP_PRESHUFFLE_SCALES"),
        "accumulation_form": args.accumulation_form,
        "python": sys.version,
        "aiter_commit": subprocess.check_output(
            ["git", "-C", "/home/leling/aiter", "rev-parse", "HEAD"], text=True
        ).strip(),
        "triton_commit": subprocess.check_output(
            ["git", "-C", "/home/leling/triton", "rev-parse", "HEAD"], text=True
        ).strip(),
        "variants": {},
    }
    print(json.dumps(result), flush=True)
    t0 = time.monotonic()
    x, w, _, xs, ws, _, _, _, out = generate_gemm_afp4wfp4_inputs(M, N, K, torch.bfloat16)
    xsh = shuffle_scale(xs).view(M // 32, K)
    wsh = shuffle_scale(ws).view(N // 32, K)
    wpacked = shuffle_weight(w).reshape(N // 16, K // 2 * 16)
    config, _ = get_gemm_config("GEMM-AFP4WFP4", M, N, K)
    packed_config, _ = get_gemm_config("GEMM-AFP4WFP4_PRESHUFFLED", M, N, K)
    print(f"Inputs ready in {time.monotonic() - t0:.2f}s", flush=True)
    rows = torch.linspace(0, M - 1, min(M, 32), device=x.device).long()
    cols = torch.linspace(0, N - 1, min(N, 32), device=x.device).long()
    ref = run_torch(x[rows], w[cols], xs[rows], ws[cols], torch.bfloat16)
    calls = {
        "unshuffled": lambda: gemm_afp4wfp4(x, w, xs, ws, y=out, config=copy.deepcopy(config)),
        "scales_only_same_config": lambda: gemm_afp4wfp4_preshuffled_scales(
            x, w, xsh, wsh, y=out, config=copy.deepcopy(config)),
        "weights_and_scales_default_config": lambda: gemm_afp4wfp4_preshuffle(
            x, wpacked, xsh, wsh, y=out, config=copy.deepcopy(packed_config)),
    }
    configs = {
        name: packed_config if name == "weights_and_scales_default_config" else config
        for name in calls
    }
    if args.include_gluon:
        gluon_config, _ = get_gemm_config("GEMM-AFP4WFP4", M, N, K, backend="gluon")
        calls["gluon_default_config"] = lambda: gemm_afp4wfp4(
            x, w, xs, ws, y=out, backend="gluon", config=copy.deepcopy(gluon_config)
        )
        configs["gluon_default_config"] = gluon_config
    rejected = []
    for name, fn in calls.items():
        fn()
        torch.cuda.synchronize()
        try:
            torch.testing.assert_close(out[rows[:, None], cols[None, :]], ref, rtol=0.02, atol=1.0)
        except AssertionError as exc:
            result["variants"][name] = {
                "config": configs[name], "correctness": "FAILED", "error": str(exc)
            }
            rejected.append(name)
            print(f"Correctness failed for {name}: {exc}", flush=True)
            continue
        result["variants"][name] = {
            "config": configs[name],
            "correctness": "32x32 sampled output entries checked against decoded FP32 PyTorch matmul",
            "milliseconds": [],
        }
        print(f"Compiled and checked {name}", flush=True)
    for name in rejected:
        del calls[name]
    # Rotate order to reduce bias from temperature and clock changes.
    names = list(calls)
    for repeat in range(args.rounds):
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        for name in order:
            ms = triton.testing.do_bench(calls[name], warmup=100, rep=500, return_mode="median")
            result["variants"][name]["milliseconds"].append(ms)
            print(f"round={repeat} {name}: {ms:.4f} ms, {2 * M * N * K / ms * 1e-9:.1f} TFLOP/s", flush=True)
        args.output.write_text(json.dumps(result, indent=2))
    for value in result["variants"].values():
        if "milliseconds" not in value:
            continue
        value["median_ms"] = statistics.median(value["milliseconds"])
        value["tflops"] = 2 * M * N * K / value["median_ms"] * 1e-9
        value["best_round_ms"] = min(value["milliseconds"])
        value["best_round_tflops"] = 2 * M * N * K / value["best_round_ms"] * 1e-9
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
