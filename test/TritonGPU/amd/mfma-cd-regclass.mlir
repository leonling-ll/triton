// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" | FileCheck %s

// With amdg.cd_regclass (set by the `cd_regclass` argument of Gluon's AMD
// `mfma`), each MFMA tile's accumulator is wrapped in an empty inline asm whose
// output is tied to its input: right before the tile's first MFMA (C) and right
// after its last MFMA (D). A constant accumulator is not pinned on C.

// CHECK-LABEL: mfma_cd_regclass_agpr
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_cd_regclass_agpr(%arg0: tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                                        %arg1: tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
                                        %arg2: tensor<16x16xf32, #mma>) {
    // Two K-steps: C pin, MFMA, MFMA, D pin.
    // CHECK: %[[C:.*]] = llvm.inline_asm asm_dialect = att "", "=a,0" {{.*}} : (vector<4xf32>) -> vector<4xf32>
    // CHECK-NEXT: %[[D0:.*]] = rocdl.mfma.f32.16x16x32.f16 {{.*}}%[[C]]
    // CHECK-NEXT: %[[D1:.*]] = rocdl.mfma.f32.16x16x32.f16 {{.*}}%[[D0]]
    // CHECK-NEXT: llvm.inline_asm asm_dialect = att "", "=a,0" %[[D1]] : (vector<4xf32>) -> vector<4xf32>
    %dot = tt.dot %arg0, %arg1, %arg2 {amdg.cd_regclass = "a"} : tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: mfma_cd_regclass_constant_acc
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_cd_regclass_constant_acc(%arg0: tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                                                %arg1: tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>) {
    // No C pin on a zero accumulator; D is still pinned.
    // CHECK-NOT: llvm.inline_asm
    // CHECK: %[[D0:.*]] = rocdl.mfma.f32.16x16x32.f16
    // CHECK-NEXT: %[[D1:.*]] = rocdl.mfma.f32.16x16x32.f16 {{.*}}%[[D0]]
    // CHECK-NEXT: llvm.inline_asm asm_dialect = att "", "=a,0" %[[D1]] : (vector<4xf32>) -> vector<4xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %dot = tt.dot %arg0, %arg1, %cst {amdg.cd_regclass = "a"} : tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: mfma_no_cd_regclass
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_no_cd_regclass(%arg0: tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                                      %arg1: tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
                                      %arg2: tensor<16x16xf32, #mma>) {
    // CHECK-NOT: llvm.inline_asm
    // CHECK: rocdl.mfma.f32.16x16x32.f16
    // CHECK: rocdl.mfma.f32.16x16x32.f16
    // CHECK-NOT: llvm.inline_asm
    // CHECK: llvm.return
    %dot = tt.dot %arg0, %arg1, %arg2 : tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// A 32x32 accumulator is pinned as one 16-register tuple.
// CHECK-LABEL: mfma_cd_regclass_vgpr_32x32
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_cd_regclass_vgpr_32x32(%arg0: tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                                              %arg1: tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
                                              %arg2: tensor<32x32xf32, #mma>) {
    // CHECK: %[[C:.*]] = llvm.inline_asm asm_dialect = att "", "=v,0" {{.*}} : (vector<16xf32>) -> vector<16xf32>
    // CHECK-NEXT: %[[D:.*]] = rocdl.mfma.f32.32x32x16.f16 {{.*}}%[[C]]
    // CHECK-NEXT: llvm.inline_asm asm_dialect = att "", "=v,0" %[[D]] : (vector<16xf32>) -> vector<16xf32>
    %dot = tt.dot %arg0, %arg1, %arg2 {amdg.cd_regclass = "v"} : tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

// Scaled MFMA (mfma_scaled): the same pins around each tile's K-steps; the zero
// accumulator is not pinned on C.
// CHECK-LABEL: mfma_scaled_cd_regclass_agpr
#linear = #ttg.linear<{register = [[0, 4], [32, 0], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 0], [16, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 4], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [32, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 128], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_scaled_cd_regclass_agpr(%arg0: tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<256x8xi8, #linear>, %arg2: tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg3: tensor<256x8xi8, #linear1>) {
    // CHECK-NOT: llvm.inline_asm
    // CHECK: %[[D0:.*]] = rocdl.mfma.scale.f32.16x16x128.f8f6f4
    // CHECK-NEXT: %[[D1:.*]] = rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}%[[D0]]
    // CHECK-NEXT: llvm.inline_asm asm_dialect = att "", "=a,0" %[[D1]] : (vector<4xf32>) -> vector<4xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %dots = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e2m1 rhs = e2m1 {amdg.cd_regclass = "a", fastMath = false} : tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear1> -> tensor<256x256xf32, #mma>
    tt.return
  }
}

// -----

// Two-step (ping-pong) scaled MFMA lowers one K step per pass, so each pass's
// MFMA is pinned. The first pass reads the zero accumulator (no C pin); the
// second pass reads the first pass's result.
// CHECK-LABEL: mfma_scaled_2step_cd_regclass
#linear = #ttg.linear<{register = [[0, 4], [32, 0], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 0], [16, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 4], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [32, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 128], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_scaled_2step_cd_regclass(%arg0: tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<256x8xi8, #linear>, %arg2: tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg3: tensor<256x8xi8, #linear1>) {
    // CHECK-NOT: llvm.inline_asm
    // CHECK: %[[P1:.*]] = rocdl.mfma.scale.f32.16x16x128.f8f6f4
    // CHECK-NEXT: llvm.inline_asm asm_dialect = att "", "=a,0" %[[P1]] : (vector<4xf32>) -> vector<4xf32>
    // CHECK: rocdl.s.barrier
    // CHECK: %[[C2:.*]] = llvm.inline_asm asm_dialect = att "", "=a,0" {{.*}} : (vector<4xf32>) -> vector<4xf32>
    // CHECK-NEXT: %[[P2:.*]] = rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}%[[C2]]
    // CHECK-NEXT: llvm.inline_asm asm_dialect = att "", "=a,0" %[[P2]] : (vector<4xf32>) -> vector<4xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %dots = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e2m1 rhs = e2m1 {amdg.cd_regclass = "a", fastMath = false, pingpong_2step} : tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear1> -> tensor<256x256xf32, #mma>
    tt.return
  }
}

// -----

// Visit each 2x2 group of MFMA tiles before moving to the next group.
// This keeps A/B operand reuse close even when C/D register-class pins and
// the two-step pingpong barriers constrain backend scheduling.
// CHECK-LABEL: mfma_scaled_2step_tiled_cd_regclass
#linear = #ttg.linear<{register = [[0, 4], [16, 0], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 0], [32, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 4], [16, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[32, 0], [64, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 128], isTransposed = true, tilesPerWarp = [2, 2]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_scaled_2step_tiled_cd_regclass(%arg0: tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<256x8xi8, #linear>, %arg2: tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg3: tensor<256x8xi8, #linear1>) {
    // Check the byte permutation itself and tie each permuted word to MFMA.
    // A and B both change from [r0k0, r0k1, r1k0, r1k1] to
    // [r0k0, r1k0, r0k1, r1k1]; r is M for A and N for B.
    // CHECK: %[[A_BYTES:.*]] = llvm.shufflevector {{.*}} [0, 2, 1, 3] : vector<4xi8>
    // CHECK-NEXT: %[[SA:.*]] = llvm.bitcast %[[A_BYTES]] : vector<4xi8> to i32
    // CHECK: llvm.extractvalue %arg3[0]
    // CHECK: %[[B_BYTES:.*]] = llvm.shufflevector {{.*}} [0, 2, 1, 3] : vector<4xi8>
    // CHECK-NEXT: %[[SB:.*]] = llvm.bitcast %[[B_BYTES]] : vector<4xi8> to i32
    // CHECK-NOT: llvm.inline_asm
    // CHECK: %[[P1:.*]] = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %[[B0:[^,]+]], %[[A0:[^,]+]], {{.*}}, fp4_e2m1, fp4_e2m1, 0, %[[SB]], 0, %[[SA]]
    // CHECK-NEXT: llvm.inline_asm asm_dialect = att "", "=a,0" %[[P1]] : (vector<4xf32>) -> vector<4xf32>
    // CHECK-NOT: rocdl.mfma
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 %[[B1:[^,]+]], %[[A0]], {{.*}}, fp4_e2m1, fp4_e2m1, 1, %[[SB]], 0, %[[SA]]
    // CHECK-NOT: rocdl.mfma
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 %[[B0]], %[[A1:[^,]+]], {{.*}}, fp4_e2m1, fp4_e2m1, 0, %[[SB]], 1, %[[SA]]
    // CHECK-NOT: rocdl.mfma
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 %[[B1]], %[[A1]], {{.*}}, fp4_e2m1, fp4_e2m1, 1, %[[SB]], 1, %[[SA]]
    // CHECK: rocdl.s.barrier
    // CHECK: %[[C2:.*]] = llvm.inline_asm asm_dialect = att "", "=a,0" {{.*}} : (vector<4xf32>) -> vector<4xf32>
    // CHECK-NEXT: %[[P2:.*]] = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %[[B0K1:[^,]+]], %[[A0K1:[^,]+]], %[[C2]], fp4_e2m1, fp4_e2m1, 2, %[[SB]], 2, %[[SA]]
    // CHECK-NEXT: llvm.inline_asm asm_dialect = att "", "=a,0" %[[P2]] : (vector<4xf32>) -> vector<4xf32>
    // CHECK-NOT: rocdl.mfma
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 %[[B1K1:[^,]+]], %[[A0K1]], {{.*}}, fp4_e2m1, fp4_e2m1, 3, %[[SB]], 2, %[[SA]]
    // CHECK-NOT: rocdl.mfma
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 %[[B0K1]], %[[A1K1:[^,]+]], {{.*}}, fp4_e2m1, fp4_e2m1, 2, %[[SB]], 3, %[[SA]]
    // CHECK-NOT: rocdl.mfma
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 %[[B1K1]], %[[A1K1]], {{.*}}, fp4_e2m1, fp4_e2m1, 3, %[[SB]], 3, %[[SA]]
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %dots = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e2m1 rhs = e2m1 {amdg.cd_regclass = "a", fastMath = false, pingpong_2step} : tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear1> -> tensor<256x256xf32, #mma>
    tt.return
  }
}

// -----

// Ordinary dots, including a pipelined epilogue, retain K-first scale
// packing and linear M/N traversal. The warp tile layout alone must not
// enable the two-step pingpong schedule.
// CHECK-LABEL: mfma_scaled_tiled_k_inner_cd_regclass
#linear = #ttg.linear<{register = [[0, 4], [16, 0], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 0], [32, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 4], [16, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[32, 0], [64, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 128], isTransposed = true, tilesPerWarp = [2, 2]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_scaled_tiled_k_inner_cd_regclass(%arg0: tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<256x8xi8, #linear>, %arg2: tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg3: tensor<256x8xi8, #linear1>) {
    // CHECK-NOT: llvm.shufflevector
    // K stays innermost. Both bytes for one (M, N) accumulator precede
    // the next N tile; packing stays K-first even for tilesPerWarp=[2,2].
    // CHECK: %[[P0:.*]] = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %[[B0:[^,]+]], %[[A0:[^,]+]], {{.*}}, fp4_e2m1, fp4_e2m1, 0, %[[SB0:[^,]+]], 0, %[[SA0:[^ ]+]]
    // CHECK-NEXT: %[[P1:.*]] = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %[[B0K1:[^,]+]], %[[A0K1:[^,]+]], %[[P0]], fp4_e2m1, fp4_e2m1, 1, %[[SB0]], 1, %[[SA0]]
    // CHECK-NEXT: llvm.inline_asm asm_dialect = att "", "=a,0" %[[P1]] : (vector<4xf32>) -> vector<4xf32>
    // CHECK-NOT: rocdl.mfma
    // CHECK: %[[P2:.*]] = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %[[B1:[^,]+]], %[[A0]], {{.*}}, fp4_e2m1, fp4_e2m1, 2, %[[SB0]], 0, %[[SA0]]
    // CHECK-NEXT: %[[P3:.*]] = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %[[B1K1:[^,]+]], %[[A0K1]], %[[P2]], fp4_e2m1, fp4_e2m1, 3, %[[SB0]], 1, %[[SA0]]
    // CHECK-NEXT: llvm.inline_asm asm_dialect = att "", "=a,0" %[[P3]] : (vector<4xf32>) -> vector<4xf32>
    // CHECK-NOT: rocdl.mfma
    // Visit N=2 with the same A operand, rather than advancing to M=1.
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{[^,]+}}, %[[A0]], {{.*}}, fp4_e2m1, fp4_e2m1, 0, {{[^,]+}}, 0, %[[SA0]]
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %dots = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e2m1 rhs = e2m1 {amdg.cd_regclass = "a", fastMath = false} : tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear1> -> tensor<256x256xf32, #mma>
    tt.return
  }
}
