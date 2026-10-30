# Cached binary and AGPR allocation experiments (2026-10-29)

All measurements below replay the saved/assembled HSACO directly through
`CompiledKernel`; the kernel is not rebuilt during timing. This isolates
machine-code differences on one MI350 GPU (`HIP_VISIBLE_DEVICES=2`, PCI
`0000:68:00.0`, gfx950, 256 CUs). No GPU clock or power setting was changed.

Logical GEMM shape: M=51200, N=106496, K=16384. Each experiment uses identical
input/output buffers for its variants. Every binary passed a 32x32 output-sample
comparison against decoded-input FP32 PyTorch matmul, with BF16 output and
rtol=0.02/atol=1.0. This is sampled correctness, not a full-output check.

Timing: seven rounds, variant order rotated each round, Triton `do_bench`
warmup=100 ms and rep=500 ms, median within each round and median across rounds.
Numbers below are TFLOP/s, with the conventional 2*M*N*K operation count.

| Saved binary | Unshuffled ms / TFLOP/s | Scale-shuffled ms / TFLOP/s |
|---|---:|---:|
| Oct 27 cache | 53.431 / 3343.9 | 55.513 / 3218.5 |
| Oct 29 earlier cache | 73.020 / 2446.9 | 72.811 / 2453.9 |
| Current baseline cache | 53.341 / 3349.6 | 54.850 / 3257.4 |

The current unshuffled binary is 36.9% faster than the earlier Oct 29 binary
on identical inputs and hardware. Kernel runtime metadata reports 230/256/224
registers and 0/13/0 spills for the three unshuffled binaries respectively.
For scale-shuffled binaries those values are 226/256/222 registers and 0/5/0
spills. Static assembly spill counts and runtime `n_spills` use different
accounting and should not be equated.

Raw results: `results.json`, `scales-results.json`; logs: `replay.log`,
`scales-replay.log`. Metadata paths and HSACO SHA256 hashes identify the exact
binary in each result. Telemetry was sampled independently during part of the
run, so `telemetry-samples.json` is supporting context, not per-variant clock
normalization. Do not infer sustained clocks from idle/inter-kernel samples.

## Controlled LLVM function-attribute experiment

`../compiler-analysis/recompile_llir.py` recompiles saved optimized LLVM IR
using the current AMD code generator, changing only the `amdgpu-agpr-alloc`
function attribute within each pair (plus the same required LLVM metadata
syntax adaptation on both old-IR variants). `replay.py --agpr` assembles those
outputs and loads them with metadata from the corresponding baseline cache.

| LLVM IR input / attribute | ms | TFLOP/s | Runtime registers / spills |
|---|---:|---:|---:|
| Current cached HSACO control | 53.369 | 3347.8 | 224 / 0 |
| Earlier Oct 29 IR, attribute absent | 83.580 | 2137.7 | 256 / 20 |
| Same earlier IR, `amdgpu-agpr-alloc=0` | 53.441 | 3343.3 | 224 / 0 |
| Current IR, existing `amdgpu-agpr-alloc=0` | 53.302 | 3352.1 | 224 / 0 |
| Same current IR, attribute removed | 86.507 | 2065.4 | 256 / 19 |

This isolates a causal benefit from setting the minimum required AGPR
allocation to zero for this kernel.
Adding the attribute to the old IR improves throughput 56.4%; removing it
from current IR reduces throughput 38.4%. These artificial same-IR comparisons
do not imply the whole historical compiler gap is exactly those percentages:
the original Oct 29 binary was produced by a different compiler build.

Raw results: `agpr-results.json`, `agpr-replay.log`. Newly assembled controlled
binaries and launch metadata are under `generated/`. All files were created
under the investigation directory; no compiler/kernel checkout or installation
was changed by this experiment.

## Reproduction

From `/home/leling/aiter`:

```sh
HIP_VISIBLE_DEVICES=2 PYTHONPATH=/home/leling/aiter \
  python /home/leling/issue-investigations/1347/cached-replay/replay.py

HIP_VISIBLE_DEVICES=2 PYTHONPATH=/home/leling/aiter \
  python /home/leling/issue-investigations/1347/cached-replay/replay.py \
  --scales --output /home/leling/issue-investigations/1347/cached-replay/scales-results.json

HIP_VISIBLE_DEVICES=2 PYTHONPATH=/home/leling/aiter \
  python /home/leling/issue-investigations/1347/cached-replay/replay.py \
  --agpr --output /home/leling/issue-investigations/1347/cached-replay/agpr-results.json
```

The replay harness expects the cached input layout: row-major FP4 data and
column-major unshuffled scales, or AITER's preshuffled scale layout. It is
purpose-built for the saved 256x256x256, eight-warp, two-stage kernels.
