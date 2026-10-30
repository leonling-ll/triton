# Issue 1347: MI350 performance gap investigation

The approximately 2400 TFLOP/s result was largely a compiler code-generation
problem. Replaying the saved slow binary and the current binary on the same
idle MI350 and identical inputs gives 2446.9 versus 3349.6 TFLOP/s (+36.9%).
The current compiler already contains the upstream LLVM fix responsible for
the recovery. An additional kernel configuration change gives a repeatable
3.5–4.1% gain, including 3282.8 to 3407.1 TFLOP/s in a sustained rotating-buffer
graph benchmark. Another idle card reached a seven-round median 3487.1 TFLOP/s
with that configuration.

Issue: https://github.com/AMD-Triton/triton-tickets/issues/1347

## Setup and correction to the earlier measurements

- Logical GEMM shape: M=51200, N=106496, K=16384, MXFP4 inputs, BF16 output.
- AITER: `ac32ca72ba333703843ab3fa5b7a37056a146813`.
- Current Triton: `732b3c4c388a0699b0f40bf9e577720b2a4ed45e`, branch
  `fix/chained-dot-extra-loads-main`, based on `cd2e01c668`.
- Current AMD code generator: `08b05a3352eeadd10eba4a25ffc19c2abb226448-2`.
- PyTorch: `2.10.0+rocm7.2.4.git3d3aa833`; Python 3.12.3.
- MI350, gfx950, 256 CUs. All eight GPUs were idle and had no KFD processes
  when the investigation began. Independent runs used HIP device 0 / PCI78,
  HIP device 1 / PCI08, and HIP device 2 / PCI68. ROCm SMI indices differ:
  these are SMI devices 3, 0, and 2 respectively.
- No clock, power, installed-package, or source-checkout changes were made.
  The existing 1000 W power cap remained in effect. Both checkouts are clean.

The saved October 29 measurement used Triton `3b0c7f081` (October 9 commit),
which is **older** than the October 27 run's `1b4a2041` (October 26 commit).
The dates of the measurements do not imply increasing compiler revisions.
This explains why the prior refresh appeared to lose performance after the
earlier successful run. The original missing dot-scaled/add combine remains
fixed; it is a separate issue from the register-allocation problem below.

## Compiler cause and controlled evidence

Seven rotated rounds replayed the original saved HSACO binaries directly,
without rebuilding them, on PCI68. Each round used 100 ms warmup and 500 ms
event measurement, reporting the median of round medians. Inputs and launch
geometry were identical within each experiment.

| Saved binary | Unshuffled TFLOP/s | Scale-shuffled TFLOP/s | Unshuffled runtime VGPRs / spills |
| --- | ---: | ---: | ---: |
| October 27 | 3343.9 | 3218.5 | 230 / 0 |
| Earlier October 29 | 2446.9 | 2453.9 | 256 / 13 |
| Current | 3349.6 | 3257.4 | 224 / 0 |

Unshuffled and shuffled columns came from separate experiments; each column
is a controlled comparison across compiler binaries, not a simultaneous
paired shuffling comparison. Every binary passed a 32x32 output sample check
against decoded FP32 PyTorch matmul converted to BF16.

The older core LLVM did not mark `llvm.amdgcn.asyncmark` and
`llvm.amdgcn.wait.asyncmark` as `nocallback`. That prevented AMDGPUAttributor
from proving the kernel's minimum required AGPR allocation was zero. The LLVM IR
therefore lacked `"amdgpu-agpr-alloc"="0"`. Later code generation used AGPR
copies and scratch spilling, and performance deteriorated sharply.

Upstream LLVM commit [`e282c27d4c4d`](https://github.com/llvm/llvm-project/commit/e282c27d4c4d)
([PR #219099](https://github.com/llvm/llvm-project/pull/219099)) adds the missing
`nocallback` properties. Triton's core LLVM update
[`1e259627c3`](https://github.com/triton-lang/triton/commit/1e259627c3)
([PR #12236](https://github.com/triton-lang/triton/pull/12236)) incorporates it.
Both the October 27 and current builds are after that update.

Two experiments establish the mechanism:

1. Compile the same saved LLVM IR with the same current AMD code generator,
   changing only `amdgpu-agpr-alloc`. Replaying those binaries gives:

   | LLVM IR and single change | TFLOP/s |
   | --- | ---: |
   | Earlier IR, attribute absent | 2137.7 |
   | Same earlier IR, add `amdgpu-agpr-alloc=0` | 3343.3 |
   | Current IR, existing attribute | 3352.1 |
   | Same current IR, remove attribute | 2065.4 |

   These numbers isolate the attribute on current codegen; they are not a
   historical LLVM bisect. All four variants passed sampled correctness.

2. Run current AMDGPUAttributor on the saved older IR. Current LLVM upgrades
   the intrinsic declarations, infers the zero-AGPR attribute, and emits an
   instruction stream identical to the current baseline: 224 VGPRs and zero
   spills. A standalone experiment removes only `nocallback` from the two
   declarations before Attributor. It then fails to infer the attribute and
   emits the same spilling stream as the earlier IR/current-codegen case:
   256 VGPRs, 33 static VGPR spills, and 80 bytes private scratch. This
   experiment does not modify the installed compiler.

The preferred compiler fix is to use a core LLVM revision containing the
intrinsic correction, as the current checkout already does. For this kernel
on an older compiler that supports LLVM function attributes, the validated
local workaround is `llvm_fn_attrs="amdgpu-agpr-alloc=0"`. This is a
kernel-specific workaround, not a recommendation to override allocation
requirements globally. Updating only the separate AMD code-generation
library does not repair optimized IR that is missing the attribute; that
pipeline does not rerun the required Attributor inference.

Detailed sources, commands and evidence are in `compiler-analysis/`; binary
replay scripts, checksums, and timings are in `cached-replay/README.md`.

## Additional kernel improvement

Use the unshuffled kernel and change only:

```json
{
  "matrix_instr_nonkdim": 32,
  "GROUP_SIZE_M": 8
}
```

Keep 256x256x256 blocks, eight warps, two stages, `waves_per_eu=1`, and no
split-K. The complete reusable override is
`kernel-tuning/unshuffled-m51200-n106496-k16384.json`.

| Paired measurement | Default TFLOP/s | Tuned TFLOP/s | Gain |
| --- | ---: | ---: | ---: |
| PCI08, seven event rounds | 3368.6 | 3487.1 | 3.52% |
| PCI78, seven event rounds | 3309.2 | 3445.1 | 4.11% |
| PCI78, three sustained graph rounds | 3282.8 | 3407.1 | 3.79% |

The graph test uses four distinct buffer allocations, ten kernels per graph,
three warmup replays, and ten measured replays per round. Variant order
alternates. AITER's input helper resets the random seed, so the four buffer
sets have identical values; they rotate physical addresses for measurement.
The PCI78 event test uses 100 ms warmup / 500 ms measurement; the PCI08 final
event test uses 100 / 800 ms. Do not interpret measurements across different
cards as controlled speedups.

The new kernel has 236 VGPRs, zero spills, and 141280 bytes LDS, versus
224 VGPRs, zero spills, and 143296 bytes LDS for the default. Its static MFMA
count changes from 128 16x16x128 instructions to 64 32x32x64 instructions,
and static `s_waitcnt` count changes from 32 to 16. It adds 32 permlane32
swaps. These static counts describe the code difference, not a dynamic
profiling attribution.

The tuning sweeps covered tiles, warps, stages, group sizes, MFMA instruction
sizes, cache policy, and scheduler options. An iterative-ILP scheduler was
another approximately 3.3% candidate; the two JSON tuning changes above
avoid relying on that experimental option. The tested MFMA scheduling hint
did not improve performance. Some larger K/stage combinations exceeded LDS.

The override is explicit for the measured shape. The repository's existing
`any` bucket covers M>256, so changing that bucket based on a single very
large shape would need additional performance coverage. This investigation
does not alter that broader dispatch policy.

## Correctness

- All timed full-shape variants passed the original 32x32 sampled checks.
- The final PCI78 comparison passed 4096 selected output positions per
  variant, repeated on each of the four buffer allocations. The positions
  repeat because the generator resets its seed; these are not 16384
  independent positions.
- The recommended MFMA32/GROUP8 config passed bit-exact **full-output**
  comparisons against decoded FP32 matmul converted to BF16 on 1024^3,
  768x1536x16384, and partial-tile 1056x1088x1024.
- Full-output reference checking at 51200x106496x16384 was not performed.

## Hardware and remaining scale-preshuffling gap

During sustained graph runs, busy telemetry samples report approximately
1000 W socket power and an active PPT power limit. The default kernel's
median across-XCD mean clock is about 1424 MHz; MFMA32/GROUP8 is about
1255 MHz. No sampled socket, VR, or HBM thermal-limit violation was active.
The tuned kernel performs more work per second despite the lower measured
clock. Clock readings are sampled telemetry, not a per-instruction profiler.

Consequently, scaling the historical MI355 results by advertised maximum
clock ratios cannot establish the remaining software gap. The MI355 runs
need actual sustained frequency/power data and the same binaries and timing
protocol for a controlled comparison. This node's maximum advertised SCLK
is 2200 MHz, substantially above the observed sustained clocks for this
workload. No power or clock limits were adjusted here.

Scale preshuffling still carries a smaller penalty. In the fresh paired
PCI78 baseline, unshuffled reaches 3325.3 versus 3191.8 TFLOP/s for shuffled
scales with the same config (4.0% slower). The focused shuffled-scale sweep
found no repeatable gain from MFMA32, MFMA scheduling, iterative ILP, or the
tested group/K changes; MFMA32 was slightly worse for this path. The
unshuffled config improvement must therefore not be applied automatically
to the scale-shuffled kernel. Scale permutation/LDS lowering and wait
scheduling remain follow-up targets. No compiler fix for that smaller
penalty is claimed.

## Reproduction and artifacts

Run the final paired event/graph comparison on an idle HIP device 0:

```sh
cd /home/leling/aiter
HIP_VISIBLE_DEVICES=0 PYTHONPATH=/home/leling/aiter \
TRITON_CACHE_DIR=/home/leling/issue-investigations/1347/final-validation/cache \
python /home/leling/issue-investigations/1347/final-validation/validate.py
```

The telemetry collector in that script is explicitly tied to PCI78, which
is HIP device 0 on this machine. Change that BDF if selecting another GPU.

To use the tuned config with existing input tensors:

```python
from aiter.ops.triton.utils.config_utils import load_config_json
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

config = load_config_json(
    "/home/leling/issue-investigations/1347/kernel-tuning/"
    "unshuffled-m51200-n106496-k16384.json"
).copy()
y = gemm_afp4wfp4(x, w, x_scales, w_scales, config=config)
```

`final-validation/results.json` records the sustained measurements and
commits; `final-validation/telemetry.jsonl` records hardware samples with
variant tags. `kernel-tuning/README.md` has the tuning and full-output test
commands. `scales-tuning/results.json` records the shuffled-scale experiments.
