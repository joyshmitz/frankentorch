# frankentorch-6h0h7 max_pool2d argmax-sidecar keep

## Target

- Bead: `frankentorch-6h0h7`
- Target row: `ft-api` Criterion `max_pool2d/grad`
- Single lever: f64 grad route computes a first-argmax sidecar during max-pool forward and scatters from it in backward, removing the full input clone plus backward window rescan.

## Baselines

- Current-tip profile baseline from `vmi1227854`: `max_pool2d/grad [96.875 ms 99.832 ms 103.23 ms]`; `avg_pool2d/grad [103.22 ms 111.44 ms 123.63 ms]`.
- Paired clean-worktree baseline on `ovh-a`: `max_pool2d/grad [88.989 ms 89.519 ms 90.245 ms]`.

## After

- Candidate after on `ovh-a`: `max_pool2d/grad [73.173 ms 74.122 ms 75.474 ms]`.
- Paired median delta: `89.519 ms -> 74.122 ms`.
- Speedup: `1.208x`; median time reduction: `17.20%`.

## Correctness And Invariants

- Forward output bits are unchanged: sidecar forward uses the same row-major scan and `>` comparison as the existing forward.
- Tie behavior is unchanged: equal maxima keep the first row-major position.
- Backward scatter order is unchanged: one deterministic per-plane loop accumulates output gradients into the saved first-argmax offsets.
- Floating point arithmetic is unchanged for forward outputs and backward additions; only the recovery of the already-selected argmax location changes.
- No RNG, shape, dtype, or error behavior changes.

## Gates

- `cargo test -p ft-kernel-cpu max_pool2d_indices_preserve_first_tie_backward -- --nocapture`: passed; final rerun on `vmi1227854`.
- `cargo test -p ft-api functional_max_pool2d_tie_order_golden -- --nocapture`: passed.
- Golden SHA-256: `d5f8182915ef53462383369da402783864d6e2e03523ab91218035932d351886`.
- `cargo test -p ft-api functional_max_pool2d_grad_matches_finite_diff -- --nocapture`: passed.
- `cargo check -p ft-kernel-cpu -p ft-api --all-targets`: passed warning-clean after removing an unused test import in `ft-api`.
- `cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`: passed.
- `cargo clippy -p ft-api --all-targets -- -D warnings`: failed on the broad pre-existing `ft-api` lint inventory; no diagnostics pointed at the new max-pool lines.
- `cargo fmt -p ft-api -p ft-kernel-cpu --check`: failed on broad pre-existing bench/example/source drift; the new ft-kernel assertion wrapping was fixed manually.
- `ubs crates/ft-api/src/lib.rs crates/ft-kernel-cpu/src/lib.rs ...`: hung for several minutes on the large Rust surface and was stopped; `git diff --check` passed.

## Score

- Impact: `3` (17.20% target-row reduction).
- Confidence: `4` (paired same-worker Criterion plus focused kernel/API/golden/finite-diff proof).
- Effort: `2`.
- Score: `3 * 4 / 2 = 6.0`, above the `>= 2.0` keep gate.

## Decision

Kept. Close `frankentorch-6h0h7` for the max-pool f64 grad argmax-sidecar lever. Re-profile before selecting the next pool primitive; do not infer that avg-pool or padded/ceil variants improved.
