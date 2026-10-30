# MXFP4 performance investigation: issue 1347

This directory records the October 29, 2026 investigation of the MI350
performance gap for the 51200 x 106496 x 16384 MXFP4 GEMM tracked by
[issue 1347](https://github.com/AMD-Triton/triton-tickets/issues/1347).

Read the [complete investigation](gap-investigation-20261029.md) for the
compiler root cause, controlled measurements, kernel improvement, and
remaining scale-preshuffling penalty.

| Evidence | Files |
| --- | --- |
| Missing LLVM `nocallback` properties cause register-allocation pressure and spills | [Compiler analysis](compiler-analysis/README.md) |
| Same-device saved-binary replay recovers 2447 to 3350 TFLOP/s | [Replay results](cached-replay/README.md) |
| MFMA32 / GROUP8 improves the current kernel by about 3.5–4.1% | [Tuning and correctness](kernel-tuning/README.md), [config override](kernel-tuning/unshuffled-m51200-n106496-k16384.json) |
| Sustained rotating-buffer graphs improve 3283 to 3407 TFLOP/s | [Results](final-validation/results.json), [script](final-validation/validate.py), [telemetry](final-validation/telemetry.jsonl) |
| Scale-preshuffling tuning remains unresolved | [Results](scales-tuning/results.json) |

The compiler correction was already present in the tested Triton checkout.
This branch adds investigation artifacts and an explicit tuning override;
it does not alter compiler behavior or AITER's default dispatch. Measurements
used Triton `732b3c4c388a0699b0f40bf9e577720b2a4ed45e` and AITER
`ac32ca72ba333703843ab3fa5b7a37056a146813`, as recorded in the reports.

The scripts and raw result files preserve their original experiment paths.
They require the AITER checkout and the matching Triton/compiler environment
described in the reports. Adjust those paths for another machine. The final
validation telemetry collector is tied to PCI `0000:78:00.0`; update it when
selecting another GPU.

Two text LLVM IR fixtures are included for reproducing the allocation
attribute and `nocallback` interventions without historical binaries.
Generated shared libraries, assembly, HSACO files, compiler caches, and
duplicated upstream source snapshots are not included. The exact historical
binary replay requires the original caches retained under
`/home/leling/issue-investigations/1347/` on the investigation machine.
The compiler report's paths to generated outputs or upstream snapshots
refer to that full local artifact set.
