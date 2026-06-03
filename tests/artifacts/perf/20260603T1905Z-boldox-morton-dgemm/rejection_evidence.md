# frankentorch-4v4t Morton-Tiled dgemm Rejection Evidence

## Target

- Crate: `ft-kernel-cpu`
- Benchmark: `cargo bench -p ft-kernel-cpu --bench gemm_bench -- matmul_f64_1024x1024x1024 --warm-up-time 1 --measurement-time 5 --sample-size 10`
- Baseline primitive: existing row-split `matrixmultiply::dgemm`
- Attempted lever: gated safe-Rust f64 GEMM using 64x64 output tiles, SIMD over four output columns, bit-reversed tile-column traversal, and row-tile parallelism

## Baseline

- Command: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench gemm_bench -- matmul_f64_1024x1024x1024 --warm-up-time 1 --measurement-time 5 --sample-size 10`
- Worker: `vmi1153651`
- Time: `[60.730 ms 85.967 ms 97.968 ms]`

## Proof

- Command: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-kernel-cpu morton_tiled_gemm_matches_reference_with_tolerance -- --nocapture`
- Worker: `vmi1149989`
- Result: passed after fixing candidate-local compile errors
- Isomorphism checked: dtype/shape/error/RNG behavior unchanged; non-target GEMM shapes stayed on the original row-split path; target square f64 output order stayed row-major; floating-point reassociation was bounded by tolerance against the existing GEMM reference.

## After

- Command: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench gemm_bench -- matmul_f64_1024x1024x1024 --warm-up-time 1 --measurement-time 5 --sample-size 10`
- Worker: `vmi1153651`
- Time: `[697.33 ms 746.29 ms 798.37 ms]`

## Decision

- Rejected. The proof passed, but the same-worker benchmark was about 8.7x slower by p50 than the fresh baseline.
- Source lever and temporary proof test were manually reverted; no regressed GEMM code is kept.
- Diagnosis: this safe-Rust tiled/SIMD layout did change the representation, but it gave up the highly optimized `matrixmultiply` microkernel and lost massively despite tile-locality.
- Next primitive: stop iterating GEMM variants in this streak and re-profile/attack a different profiler-backed hotspot. The next filed target is cache-blocked KNN query x point tiling with per-query stable top-k buffers, based on the prior `knn_search/8192x512_k8` profile and rejected scalar-square attempt.
- Score: impact -3 x confidence 4 / effort 2 = -6.0.
