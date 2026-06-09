# frankentorch-c31ar dense SVD deferred-left evidence

Date: 2026-06-09
Agent: BoldFalcon
Bead: frankentorch-c31ar
Target: `svd_f64_256x256_wellcond` in `ft-kernel-cpu`

## Change

Full-rank reduced square SVD now takes a guarded deferred-left route:

- run the existing Golub-Reinsch bidiagonal QR while accumulating singular values and `V`
- skip left-vector accumulation inside the QR sweep
- reconstruct `U` as `A * V * diag(1 / s)` through the existing safe-Rust `gemm::dgemm`
- fall back for `full_matrices`, non-square, `n < 64`, non-finite, rank-deficient, or ill-conditioned inputs

This keeps the rank-deficient square fast path ahead of the new route.

## Benchmark

Same command and worker:

`RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- 'svd_f64_256x256_wellcond|svdvals_f64_256x128|svdvals_f64_384x128' --warm-up-time 1 --measurement-time 5 --sample-size 10`

Baseline on `vmi1227854`:

- `svd_f64_256x256_wellcond`: `[323.09 ms 326.56 ms 330.54 ms]`
- `svdvals_f64_256x128`: `[6.5638 ms 6.7488 ms 7.0993 ms]`
- `svdvals_f64_384x128`: `[11.184 ms 11.654 ms 12.107 ms]`

After on `vmi1227854`:

- `svd_f64_256x256_wellcond`: `[190.61 ms 192.75 ms 194.94 ms]`
- `svdvals_f64_256x128`: `[5.9010 ms 6.4640 ms 6.7065 ms]`
- `svdvals_f64_384x128`: `[10.904 ms 11.376 ms 12.211 ms]`

Target median improved `326.56 ms -> 192.75 ms`, a `1.69x` speedup and `40.99%` lower median time.

Score: Impact `4` x Confidence `4` / Effort `2` = `8.0`; kept.

## Isomorphism proof

- Singular values remain sorted descending with the existing `total_cmp` ordering.
- No RNG or data-dependent scheduling is introduced.
- The accelerated path is limited to finite, well-conditioned, full-rank, reduced square matrices.
- Fallback preserves existing behavior for rank-deficient, non-finite, non-square, small, and `full_matrices` cases.
- `U = A * V / s` is the same SVD subspace relation used by the definition of SVD for nonzero singular values.
- Tests verify singular values against the values-only route, reconstruction, U orthogonality, Vh orthogonality, rank-deficient rejection, and 1-thread vs 4-thread bit-exact output.

## Gates

- `rch exec -- cargo test -p ft-kernel-cpu svd_deferred_left -- --nocapture`: passed via local fallback after remote refusal; 3 tests passed.
- `rch exec -- cargo test -p ft-kernel-cpu svd -- --nocapture`: passed via local fallback after remote refusal; 18 tests passed.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo check -p ft-kernel-cpu --all-targets`: passed remotely.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`: passed remotely.
- `git diff --check -- crates/ft-kernel-cpu/src/lib.rs .skill-loop-progress.md`: passed.
- `ubs crates/ft-kernel-cpu/src/lib.rs`: completed; only broad pre-existing warnings in the large kernel file.
- `cargo fmt -p ft-kernel-cpu --check`: failed on pre-existing unrelated formatting drift outside the deferred-left hunk.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo test -p ft-conformance torch_linalg_svd_numpy_subprocess_conformance -- --nocapture`: remote refused by RCH admission.
