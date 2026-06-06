# frankentorch-b39m performance report

## Target

- Bead: `frankentorch-b39m`
- Target: `[perf] ft-optim: borrow persistent gradients during optimizer step`
- Profile-backed source: `artifacts/perf/frankentorch-ftoptim-adamw-inplace-state/report.md` named the remaining persistent-gradient clone in the optimizer step path.
- Worker: `ts1`

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-optim --bench optimizer_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 10
```

Criterion intervals:

- `adamw/step_64x1024`: `[332.77 us 336.93 us 340.21 us]`
- `adam/step_64x1024`: `[348.01 us 361.14 us 388.74 us]`
- `sgd/step_64x1024`: `[207.42 us 213.32 us 224.62 us]`

## Candidate Gate

The first broad candidate routed both Adam and AdamW through the borrowed-gradient helper. That was rejected before commit because AdamW regressed:

- AdamW broad route: `336.93 us -> 357.98 us` median, rejected.
- Adam broad route: `361.14 us -> 347.22 us` median, useful but mixed with AdamW regression.

The kept lever is therefore narrower: add the borrowed-gradient tape/session helper and route only Adam through it. AdamW remains on the previously proven in-place state update path.

## Kept Result

Same command and worker after the Adam-only lever:

- `adamw/step_64x1024`: `[326.12 us 341.44 us 353.36 us]`
- `adam/step_64x1024`: `[288.78 us 306.00 us 322.39 us]`
- `sgd/step_64x1024`: `[214.67 us 224.33 us 241.10 us]`

Profile target win:

- Adam median: `361.14 us -> 306.00 us`
- Ratio: `1.180x`
- Criterion confidence intervals do not overlap for Adam.

Score:

- `Impact 4.0 x Confidence 0.90 / Effort 1.5 = 2.40`
- Keep threshold: `>= 2.0`

## Isomorphism Proof

Behavior is unchanged by construction:

- Parameter ordering is still `self.params.iter().enumerate()`.
- Missing-gradient behavior is unchanged: no persistent gradient still skips the parameter before advancing the step counter.
- Adam step-count ordering is unchanged: after a present gradient is observed, the step counter advances before parameter/state validation, matching the prior path.
- Gradient length mismatch behavior is unchanged: the candidate validates gradient length against parameter length before mutation.
- State length mismatch behavior is unchanged: first- and second-moment buffers are validated before parameter mutation.
- Floating-point expression order is unchanged inside the per-element Adam update: `g_eff`, `m`, `v`, `m_hat`, `v_hat`, and `p -= lr * m_hat / (sqrt(v_hat) + eps)` execute in the same order.
- No RNG, tie-breaking, sorting, or iteration-order behavior is introduced.
- The persistent gradient buffer is borrowed during the update and remains stored after mutation; the focused autograd test verifies this.

Golden-output proof:

- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed.

Focused RCH proof:

- `cargo check -p ft-autograd -p ft-api -p ft-optim --all-targets` passed on `ts1`.
- `cargo test -p ft-autograd borrowed_accumulated_gradient_update_preserves_gradient_storage` passed.
- `cargo test -p ft-optim adam -- --nocapture` passed 56/56 matching filtered tests.
- `cargo clippy -p ft-optim --all-targets --no-deps -- -D warnings` passed. It reports the existing dependency warning in `ft-nn`, but the clippy command exits 0.

Formatting/scan notes:

- `cargo fmt -p ft-autograd -p ft-api -p ft-optim -- --check` cannot be run via RCH because RCH refuses non-compilation commands when remote execution is required.
- Local `cargo fmt` and direct touched-file `rustfmt --check` both report broad pre-existing formatting drift in `ft-api`/`ft-autograd` and unrelated `ft-api` benches. Those broad rewrites were intentionally not applied inside this one-lever perf commit.
- `ubs` on the three touched files emitted no findings before timing out after roughly 5 minutes; the scan process was terminated and recorded in `ubs_touched_files.txt`.

## Evidence

- `baseline_optimizer_bench.txt`
- `after_borrowed_gradient_optimizer_bench.txt`
- `after_adam_only_borrowed_gradient_optimizer_bench.txt`
- `check_adam_only_borrowed_gradient.txt`
- `test_adam_only_borrowed_gradient_tape.txt`
- `test_adam_only_adam.txt`
- `clippy_ft_optim.txt`
- `golden_sha256_check.txt`
- `source_diff_adam_only.txt`
- `ubs_touched_files.txt`
