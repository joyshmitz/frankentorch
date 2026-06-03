# ft-api lgamma no-grad fast path pass 27

Bead: `frankentorch-kgs4.31`

Target: `ft-api` `tensor_gammaln` on no-grad large tensors.

Profile evidence:
- Existing perf bead `frankentorch-a30n` identified the residual hot cost: large `lgamma`/`digamma` evaluation was parallelized, but `tensor_gammaln` still paid a serial `save_for_backward` copy even when the input did not require gradients.
- Fresh rch Criterion baseline before the lever:
  - Command: `rch exec -- cargo bench -p ft-api --bench special_bench -- lgamma_1m --warm-up-time 1 --measurement-time 5 --sample-size 20`
  - Worker: `vmi1264463`
  - Result: `[83.377 ms 98.213 ms 114.27 ms]`

Lever:
- Added a no-grad / grad-disabled `tensor_gammaln` fast path matching the established `tensor_i0` pattern.
- The fast path maps `lgamma_approx` through `par_map_f64` and builds output storage from the original tensor metadata.
- Requires-grad inputs keep the existing `tensor_apply_function` body and digamma backward closure unchanged.

After benchmark:
- Command: `rch exec -- cargo bench -p ft-api --bench special_bench -- lgamma_1m --warm-up-time 1 --measurement-time 5 --sample-size 20`
- Worker: `vmi1156319`
- Result: `[44.381 ms 49.265 ms 53.195 ms]`
- p50 delta: `98.213 ms -> 49.265 ms` (`1.99x`, `49.84%` faster).
- Score: `Impact 4 * Confidence 0.7 / Effort 1 = 2.8`; keep.

Isomorphism proof:
- Ordering: pure indexed elementwise map; Rayon indexed collection preserves output index order.
- Tie-breaking: none.
- Floating point: each output element evaluates exactly one `lgamma_approx(x)`; no reductions or reassociation.
- RNG: no RNG use in the lever; benchmark input creation is outside the op body.
- Autograd: tracked inputs still use the existing `tensor_apply_function` path and digamma backward closure.
- Ledger: fast and tracked paths record the same dispatch summary pattern, `gammaln in=<id> out=<id>`.

Golden output:
- Fixture: `artifacts/optimization/golden_outputs/ft_api_lgamma_no_grad_fast_path_pass27.txt`
- sha256: `8b92a314f9cb4d9225692206b2f38ee3f5c09d7f10a5d4c66fc9876a20c294b3`
- Full-output digest in fixture:
  - fast path: `0x5dceb5a6b608b006`
  - tracked path: `0x5dceb5a6b608b006`

Validation:
- PASS: `rch exec -- cargo test -p ft-api gammaln -- --nocapture`
  - 8 passed: includes `gammaln_no_grad_fast_path_golden_summary_matches_fixture` and `gammaln_propagates_gradient_via_digamma`.
- PASS: `rch exec -- cargo check -p ft-api --all-targets`
  - Existing warning remains: unused `meta` in unrelated `ft-api` test code.
- BLOCKED by pre-existing drift: `rch exec -- cargo fmt -p ft-api --check`
  - Fails on unrelated `ft-api` source/bench formatting created outside this pass.
- BLOCKED by pre-existing drift: `rch exec -- cargo clippy -p ft-api --all-targets --no-deps -- -D warnings`
  - Fails on many unrelated `ft-api` lints after concurrent peer commits; not caused by this lever.

Concurrency note:
- The source hunk and test were incorporated into `HEAD` by concurrent shared-worktree commit `701ae270` before this evidence commit.
- This evidence commit carries the missing golden artifact and tracker closeout for `frankentorch-kgs4.31`.
