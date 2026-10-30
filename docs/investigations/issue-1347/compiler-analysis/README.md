# Issue 1347: compiler cause of the 2400 TFLOP/s result

Investigation date: 2026-10-29. Exact workload: M=51200, N=106496,
K=16384, MXFP4 GEMM, default unshuffled 256x256x256 tile, 8 warps,
2 stages, waves_per_eu=1, MFMA non-K dimension 16.

## Finding

The saved approximately 2466 TFLOP/s result was produced with Triton
`3b0c7f081a3fc8471cae4941d1c579f2348bcbca`, an **older** October 9
compiler, even though the measurement occurred October 29. The October 27
approximately 3322 TFLOP/s run used the newer October 26 commit
`1b4a2041b0e3eda43ad398636098ef9147e4663c`.

The older compiler's core LLVM lacks the async-marker `nocallback` fix.
The resulting loss of AGPR allocation inference causes register pressure
and spills. This is already fixed by the LLVM revision consumed by the
current Triton checkout (`732b3c4c38`, based on upstream `cd2e01c668`).
Fresh current code has 224 VGPRs and zero spills; the cached older code
has 256 VGPRs and nonzero scratch accesses. The workload and tile did not
need to change to recover approximately 3300 TFLOP/s.

## Exact upstream fix and mechanism

LLVM commit
[`e282c27d4c4d1c4fd8cea8d0dcb4674ad2dff02d`](https://github.com/llvm/llvm-project/commit/e282c27d4c4d1c4fd8cea8d0dcb4674ad2dff02d),
“[AMDGPU] Mark the async marker intrinsics nocallback”
([LLVM PR #219099](https://github.com/llvm/llvm-project/pull/219099)),
was committed September 3, 2026. It adds `IntrNoCallback`, `IntrNoFree`,
and `IntrWillReturn` to `llvm.amdgcn.asyncmark` and
`llvm.amdgcn.wait.asyncmark`.

The markers only mark or wait for async memory requests; they cannot
invoke arbitrary other functions. Without `nocallback`,
`AAAMDGPUMinAGPRAlloc` must assume such calls could need AGPRs. It fails
to infer `"amdgpu-agpr-alloc"="0"`. As the upstream commit explains,
`getMaxNumVectorRegs` then conservatively reserves half the vector
register file for AGPRs, reducing the architectural VGPR budget.
The zero attribute describes the minimum required AGPR allocation; it
should not be described as a blanket “disable AGPR” switch.

Triton commit
[`1e259627c3426df5372396ba0640b76f3deb6402`](https://github.com/triton-lang/triton/commit/1e259627c3426df5372396ba0640b76f3deb6402),
“[LLVM] Update LLVM to 18f9e623”
([Triton PR #12236](https://github.com/triton-lang/triton/pull/12236)),
updates core LLVM from `b010a18d2b648cab83c83967ff26b8fde11acdc6`
(August 3) to `18f9e623e4c891ba31ff817c07197707ac9bc593`, which contains
the fix. Current AMD code-generation library is
`08b05a3352eeadd10eba4a25ffc19c2abb226448-2`.

The required fix belongs to the core LLVM IR optimization stage as well
as any code generator that runs attribute inference. Updating only the
separate AMD code-generation library does not repair already-optimized
LLVM IR that lacks the allocation attribute: the codegen pipeline does
not rerun this Attributor analysis.

The exact LLVM commit JSON is saved alongside this report. The relevant
old/new source snapshots remain in the original local artifact directory;
the compact Git package uses the upstream revision links above.

## Controlled proof

### LLVM attribute inference

Start with the exact cached October 29 LLVM IR. Its Triton TTIR is
identical to the October 27 unshuffled kernel. Adapt only the alias-scope
domain metadata to LLVM 23's parser format (`i1 false`, preserving old
semantics); this common adaptation is used in both arms.

Run current `opt -passes=amdgpu-attributor`. The parser supplies current
intrinsic attributes. The result infers `amdgpu-agpr-alloc=0`. To isolate
`nocallback` from the other intrinsic attributes, the standalone
`strip_marker_nocallback.cpp` pass removes **only** `Attribute::NoCallback`
from the two marker declarations after parsing, then runs the same
Attributor. The second arm no longer infers the allocation attribute.
No Triton checkout or installed library was changed.

| Input to the same Attributor | Inferred AGPR attribute | AMDGCN VGPRs | Assembly `.vgpr_spill_count` | Private bytes/thread |
| --- | --- | ---: | ---: | ---: |
| Current marker attributes | `0` | 224 | 0 | 0 |
| Remove only `nocallback` | absent | 256 | 33 | 80 |

The first arm's assembly instruction stream is SHA256
`f0cab9d279528abaa17b57656f55739290f4accd0f8b6c6a5710b709d2a5ca81`,
identical to recompiling current baseline LLVM IR. The second arm's
instruction stream is SHA256
`80716df5c96284a65258c42f60d0787f70d9f0233a80cb1a7c343cb18eb3c72f`,
identical to recompiling the old LLVM IR without the allocation
attribute. This controls both the source kernel and code-generation
revision.

### Direct allocation-attribute intervention and GPU replay

`recompile_llir.py` compiles cached optimized IR with the current AMD LLVM
library. Adding only `amdgpu-agpr-alloc=0` to the old IR eliminates its
spills. Removing only that attribute from current IR introduces spills.
The separate cached-replay harness then loads and times those exact
binaries with identical input tensors and grid. All four binaries pass
the existing 32x32 sampled output check against a decoded FP32 reference.

| LLVM IR and intervention | Median TFLOP/s, seven rotated rounds |
| --- | ---: |
| Old IR, attribute absent | 2137.7 |
| Old IR, add attribute `0` | 3343.3 |
| Current IR, attribute retained | 3352.1 |
| Current IR, remove attribute | 2065.4 |

See `../cached-replay/agpr-results.json` for exact timings and device
details. These intervention variants are compiled with the same current
code generator; the approximately 2138 value is not expected to equal
the approximately 2466 value from the cached original compiler binary.

Replay of the three original cached binaries on the same GPU also
reproduces the older binary's slowdown. See
`../cached-replay/results.json` for the recorded comparison. This rules
out GPU availability/frequency as the explanation for the dominant
approximately 2400-to-3300 difference.

Assembly `.vgpr_spill_count` and HIP `load_binary`'s `n_spills` use
different accounting and should not be mixed. The table above labels
the assembly metric explicitly. Scratch operations are visible in the
assembly of the slow arms and absent in the fast arms.

## Reproduction commands

From `/home/leling/issue-investigations/1347/compiler-analysis`:

```sh
c++ -std=c++17 -fPIC -shared -fno-rtti \
  -I/root/.triton/llvm/llvm-18f9e623-ubuntu-x64-2/include \
  strip_marker_nocallback.cpp -o strip_marker_nocallback.so

/root/.triton/llvm/llvm-18f9e623-ubuntu-x64-2/bin/opt -S \
  -passes=amdgpu-attributor oldir-newcodegen-agpr-unchanged.llir \
  -o oldir-attributor.llir

/root/.triton/llvm/llvm-18f9e623-ubuntu-x64-2/bin/opt -S \
  -load-pass-plugin=./strip_marker_nocallback.so \
  -passes='strip-marker-nocallback,amdgpu-attributor' \
  oldir-newcodegen-agpr-unchanged.llir \
  -o oldir-attributor-no-callback.llir

PYTHONPATH=/home/leling/triton/python python recompile_llir.py \
  --source oldir-attributor.llir --output-prefix oldir-attributor-codegen
PYTHONPATH=/home/leling/triton/python python recompile_llir.py \
  --source oldir-attributor-no-callback.llir \
  --output-prefix oldir-attributor-no-callback-codegen
```

`recompile_llir.py --agpr 0` and `--agpr remove` reproduce the direct
attribute interventions. JSON files beside each assembly record source,
codegen revision, instruction hash, resource metrics, and instruction
counts.

## Fix recommendation and remaining scope

Use a Triton build whose **core LLVM includes e282c27d4c4d**. For an older
compiler branch that cannot take the LLVM bump, backport the two-intrinsic
attribute fix and its existing LLVM regression test. A focused kernel
workaround is `llvm_fn_attrs="amdgpu-agpr-alloc=0"` for this known kernel;
the production recommendation is correct inference rather than applying
the attribute globally to arbitrary kernels.

The remaining roughly four-percent same-configuration scale-preshuffle
penalty is a different problem. Current preshuffled assembly still has
extra `v_perm_b32` instructions and different LDS scale accesses. The
scale layout's low two register bits are reversed before packing values
for MFMA byte selection, so layout/packing/op_sel coordination is a
specific next target. No causal fix for that smaller penalty is claimed
by this report.
