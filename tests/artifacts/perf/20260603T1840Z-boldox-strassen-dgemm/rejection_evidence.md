# frankentorch-1e7c Recursive Strassen dgemm Rejection Evidence

## Target

- Crate: `ft-kernel-cpu`
- Benchmark: `cargo bench -p ft-kernel-cpu --bench gemm_bench -- matmul_f64_1024x1024x1024 --warm-up-time 1 --measurement-time 5 --sample-size 10`
- Baseline primitive: existing row-split `matrixmultiply::dgemm`
- Attempted lever: recursive Strassen-style f64 square GEMM for power-of-two large matrices, with cutoff fallback to existing exact GEMM

## Baseline

- Command: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench gemm_bench -- matmul_f64_1024x1024x1024 --warm-up-time 1 --measurement-time 5 --sample-size 10`
- Worker: `vmi1153651`
- Time: `[31.134 ms 36.208 ms 40.593 ms]`

## Proof

- Command: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-kernel-cpu strassen_square_matches_reference_with_tolerance -- --nocapture`
- Worker: `vmi1293453`
- Result: passed
- Isomorphism checked: dtype/shape/error/RNG behavior unchanged; row-major output order preserved; floating-point reassociation accepted only within the test tolerance against the existing GEMM reference.

## After

- Command: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench gemm_bench -- matmul_f64_1024x1024x1024 --warm-up-time 1 --measurement-time 5 --sample-size 10`
- Worker: `vmi1293453`
- Time: `[39.628 ms 44.540 ms 48.852 ms]`

## Decision

- Rejected. The Strassen pilot passed the numerical proof but regressed by about 1.23x by p50 against the fresh baseline.
- Source lever and temporary proof test were manually reverted; no regressed GEMM code is kept.
- Diagnosis: the algorithmic flop reduction is being eaten by allocation, quadrant copy, and temporary matrix traffic.
- Next primitive: workspace-backed Winograd/Strassen schedule with preallocated scratch buffers and fused quadrant add/sub packing, targeting at least 1.25x over the current 1024x1024 baseline.
- Score: impact -1 x confidence 4 / effort 2 = -2.0.
