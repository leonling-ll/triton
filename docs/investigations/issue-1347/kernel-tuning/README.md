# Issue 1347 kernel tuning, 2026-10-29

The best tested unshuffled config uses 32x32 MFMA instructions and
`GROUP_SIZE_M=8`. On the current compiler it reached a median 3487.1 TFLOP/s,
3.52% above the paired default config. These are MI350 measurements of the
logical shape M=51200, N=106496, K=16384. They do not measure a historical
MI355 regression.

The experiment uses public AITER wrappers. No source checkout, installed
package, GPU clock, or power setting was changed. HIP device 1 was idle
before the run. It maps to PCI bus 0x08 and has gfx950, 256 CUs.

- AITER: `ac32ca72ba333703843ab3fa5b7a37056a146813`.
- Triton: `732b3c4c388a0699b0f40bf9e577720b2a4ed45e`.
- Both source checkouts were clean after the runs.
- All timings use BF16 output, eight warps, two stages, no split-K, and
  256x256x256 blocks unless the sweep config specifies a change.

## Paired final measurements

Seven rounds use 100 ms warmup and 800 ms measurement per variant. The
variant order reverses each round. The table uses the median of the seven
round medians. The default is also measured again during each round.

| Variant | Median ms | TFLOP/s | Speedup |
| --- | ---: | ---: | ---: |
| Default, MFMA16, GROUP4 | 53.0401 | 3368.6 | 1.000 |
| MFMA32, GROUP4 | 51.9730 | 3437.8 | 1.021 |
| MFMA32, GROUP8 | 51.2372 | 3487.1 | 1.035 |
| MFMA16, GROUP4, iterative-ilp scheduler | 51.3428 | 3480.0 | 1.033 |

The first sweep had a best round of 3518 TFLOP/s with MFMA32/GROUP4.
That transient best is not the sustained result. Timings slowly increased
across the longer run for every variant. The paired improvement remained.

The first 24 configs vary blocks, warps, stages, MFMA size, group size,
waves per EU, and cache policy. Smaller blocks reduced performance. Larger
K blocks with two stages exceeded shared-memory capacity. The second
sweep adds scheduler controls and checks MFMA32 settings. Neither
`schedule_hint="mfma-schedule"` nor the minreg scheduler improved the best
config. The iterative-ilp option is an alternative experiment, not part of
the recommended config override.

## Correctness and generated code

Every timed full-size candidate passed a 32x32 grid of output samples
against decoded FP32 PyTorch matmul converted to BF16. The recommended
MFMA32/GROUP8 config also passed full-output checks for 1024x1024x1024,
768x1536x16384, and the partial-tile case 1056x1088x1024. All three full
outputs matched the BF16 reference exactly. Full-size full-output checking
was not performed.

| Code property | Default | MFMA32/GROUP8 |
| --- | ---: | ---: |
| VGPRs | 224 | 236 |
| AGPRs | 0 | 0 |
| Scratch bytes | 0 | 0 |
| VGPR / SGPR spills | 0 / 0 | 0 / 0 |
| LDS bytes | 143296 | 141280 |
| Static scaled-MFMA instructions | 128 (16x16x128) | 64 (32x32x64) |
| Static s_waitcnt instructions | 32 | 16 |

MFMA32 also has 32 static permlane32 swap instructions. These are static
whole-function counts. They explain code differences but do not establish
the dynamic cause of the measured speedup.

## Reuse

`unshuffled-m51200-n106496-k16384.json` is a config override for the exact
measured shape. It changes only MFMA size and M group size from the existing
config. Load it with AITER's `load_config_json` and pass a copy to
`gemm_afp4wfp4(..., config=config)`.

The normal wrapper passes byte K to its internal `_get_config`, which
doubles it before calling `get_gemm_config`. A specialized repository file
for this case therefore has `K=16384` in its name. The current specialized
file uses an `any` bucket for M >256. Replacing that bucket would change
other M values without performance measurements. This experiment supplies
an explicit override instead of changing that broad bucket.

From `/home/leling/aiter`, rerun the paired comparison:

```sh
HIP_VISIBLE_DEVICES=1 PYTHONPATH=/home/leling/aiter \
TRITON_CACHE_DIR=/home/leling/issue-investigations/1347/kernel-tuning/cache \
python /home/leling/issue-investigations/1347/kernel-tuning/sweep.py \
  --configs /home/leling/issue-investigations/1347/kernel-tuning/finalists.json \
  --output /home/leling/issue-investigations/1347/kernel-tuning/recheck.json \
  --rounds 7 --rep 800
```

`sweep.py --configs` also accepts the single-config JSON override. Use
`validate.py` to repeat the full-output checks. `summary.json` records the
final medians and commits. `candidate-metadata.json` records assembly
properties and cache paths. `sweep1.json` and `sweep2.json` contain the full
exploration, including resource failures. `finalists-results.json` contains
all final round timings.

To benchmark the recommended override with a baseline check on an available
HIP device 1:

```sh
HIP_VISIBLE_DEVICES=1 PYTHONPATH=/home/leling/aiter \
TRITON_CACHE_DIR=/home/leling/issue-investigations/1347/kernel-tuning/cache \
python /home/leling/issue-investigations/1347/kernel-tuning/sweep.py \
  --configs /home/leling/issue-investigations/1347/kernel-tuning/unshuffled-m51200-n106496-k16384.json \
  --output /home/leling/issue-investigations/1347/kernel-tuning/override-recheck.json \
  --rounds 7 --rep 800
```

To use the override with existing tensors of the measured shape:

```python
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton.utils.config_utils import load_config_json

config = load_config_json(
    "/home/leling/issue-investigations/1347/kernel-tuning/"
    "unshuffled-m51200-n106496-k16384.json"
)
assert x.shape == (51200, 8192) and w.shape == (106496, 8192)
result = gemm_afp4wfp4(x, w, x_scales, w_scales, y=out, config=config.copy())
```

The shape assertions use packed byte K=8192. The logical GEMM K is 16384.
