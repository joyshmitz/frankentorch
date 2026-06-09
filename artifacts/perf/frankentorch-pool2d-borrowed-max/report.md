# frankentorch-b03fn max_pool2d borrowed-input reject

## Target

- Bead: `frankentorch-b03fn`
- Target row: `ft-api` Criterion `max_pool2d/grad`
- Worker requirement: same-worker RCH comparison
- Single lever tested: route f64 `functional_max_pool2d` grad through `tensor_apply_function_f64_borrowed_inputs` so backward borrows the immutable input from the tape instead of saving a full cloned input.

## Baseline

- Command: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-api --bench ops_bench -- pool2d/grad --sample-size 10 --warm-up-time 1 --measurement-time 3`
- Worker: `vmi1227854`
- `max_pool2d/grad`: `[96.875 ms 99.832 ms 103.23 ms]`
- `avg_pool2d/grad`: `[103.22 ms 111.44 ms 123.63 ms]`

## Candidate

- Command: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -p ft-api --bench ops_bench -- max_pool2d/grad --sample-size 10 --warm-up-time 1 --measurement-time 3`
- Worker: `vmi1227854`
- `max_pool2d/grad`: `[101.97 ms 108.28 ms 117.16 ms]`
- Median delta: `99.832 ms -> 108.28 ms`, `0.922x` throughput / `8.46%` slower.

## Correctness Evidence During Probe

- Golden tie-order test passed through RCH.
- Golden line SHA-256: `d5f8182915ef53462383369da402783864d6e2e03523ab91218035932d351886`
- Existing finite-difference `functional_max_pool2d_grad_matches_finite_diff` passed through RCH on `vmi1227854`.
- `cargo check -p ft-api --all-targets` passed warning-clean during the candidate probe after removing one local unused test import.
- `cargo clippy -p ft-api --all-targets -- -D warnings` failed on the broad pre-existing `ft-api` clippy inventory; no new source was kept.

## Decision

- Score: `0.0`, below the `>= 2.0` keep gate.
- Verdict: rejected; source/test hunks were manually removed and `crates/ft-api/src/lib.rs` is clean against `HEAD`.
- Next route: do not retry borrowed-input-only tape plumbing. The deeper pool primitive is an argmax-sidecar max-pool forward/backward path or a reverse-mapped backward scatter that removes the backward window rescan while preserving first-tie semantics.
