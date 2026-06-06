# frankentorch-e82f: batched det Rayon wrapper rejected

## Target

- Bead: `frankentorch-e82f` (`[perf][no-gaps] ft-api: batched linalg`)
- Profile-backed hotspot: current torch-style batched determinant users must loop over independent 2-D matrices because `tensor_linalg_det` rejects `[*, n, n]`.
- Candidate primitive: one batched `tensor_linalg_det` wrapper over `[batch, n, n]`, computing each matrix determinant independently with `rayon` over matrix chunks.

## Baseline

Baseline command on `ts1`:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- batched_linalg/det_loop --warm-up-time 1 --measurement-time 5 --sample-size 10
```

Baseline current workaround:

- `batched_linalg/det_loop_256x16x16`: `[440.43 us 445.77 us 453.98 us]`

## Candidate Result

Candidate A/B command on `ts1`:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- batched_linalg/det --warm-up-time 1 --measurement-time 5 --sample-size 10
```

Same-worker after:

- loop confirmation: `[476.64 us 489.85 us 510.07 us]`
- candidate batched wrapper: `[696.29 us 726.36 us 769.34 us]`

Verdict: rejected. The candidate regressed against the original loop baseline (`445.77 us -> 726.36 us`) and the same-run loop confirmation (`489.85 us -> 726.36 us`).

Score: `Impact 0.61 x Confidence 0.95 / Effort 1.0 = 0.58`, below the `2.0` keep gate.

## Proof Status

- Candidate source hunk is retained only as `candidate_source_diff_rejected.txt`.
- Candidate focused tests passed before removal: `det_batched_matches_independent_2d_results`, `det_batched_backward_matches_per_matrix_jacobi_formula`, and `det_batched_rejects_non_square_trailing_dims`.
- Final source diff is empty after rejection (`final_source_diff.txt`).
- Golden SHA-256 check passed after removal (`golden_sha256_check.txt`).
- `git diff --check` passed (`git_diff_check.txt`).

## Next Primitive

The rejected lever shows that a naive per-matrix Rayon wrapper is the wrong abstraction for small batched determinants. The next attack should be a deeper kernel-level batched LU/det primitive with an amortized workspace, serial chunking threshold, and optional coarse batch parallelism only above a measured break-even point. Target ratio: at least `1.5x` over the loop workaround on the same `256x16x16` family before expanding to batched `inv`/`solve`/`cholesky`.
