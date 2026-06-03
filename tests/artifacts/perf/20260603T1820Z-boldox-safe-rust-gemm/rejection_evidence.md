# frankentorch-5teh Safe-Rust GEMM Rejection Evidence

## Target

- Crate: `ft-kernel-cpu`
- Benchmark: `cargo bench -p ft-kernel-cpu --bench gemm_bench -- matmul_f64_1024x1024x1024 --warm-up-time 1 --measurement-time 5 --sample-size 10`
- Baseline primitive: existing row-split `matrixmultiply::dgemm`
- Attempted lever: owned safe-Rust 4x4 `wide::f64x4` register-tiled `dgemm_block` without packed panels

## Baseline

- Command: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench gemm_bench -- matmul_f64_1024x1024x1024 --warm-up-time 1 --measurement-time 5 --sample-size 10`
- Worker: `vmi1156319`
- Time: `[24.045 ms 25.633 ms 28.166 ms]`

## Proof

- Command: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-kernel-cpu gemm_row_split_matches_single_bit_exact -- --nocapture`
- Worker: `vmi1149989`
- Result: passed
- Isomorphism checked: per-output `k` accumulation order remained deterministic and row splitting did not change output bits for the attempted safe dgemm path.

## After

- Command: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench gemm_bench -- matmul_f64_1024x1024x1024 --warm-up-time 1 --measurement-time 5 --sample-size 10`
- Worker: `vmi1149989`
- Time: `[72.253 ms 79.666 ms 89.347 ms]`

## Decision

- Rejected. The no-pack register tile is roughly 3.1x slower by p50 than the existing baseline despite passing the row-split proof.
- Source lever was manually reverted; no regressed GEMM code is kept.
- Diagnosis: the attempted kernel rereads the full B stream for every small row tile and does not provide the packed-panel/cache-blocking primitive required to compete with the existing microkernel.
- Next primitive: packed KC/NC B panels plus A micro-panels feeding an MR x NR safe-Rust register kernel, with panel reuse as the primary profile-backed lever.
- Score: impact -2 x confidence 4 / effort 1 = -8.0.
