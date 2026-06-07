# frankentorch-t1vg rejected: borrowed Linear grad inputs

Date: 2026-06-07
Agent: RubyLotus
Skill loop: `repeatedly-apply-skill` over `extreme-software-optimization`

## Target

`linear_train/hidden/2048` in `crates/ft-api/benches/ops_bench.rs`.

The profiled source target was `functional_linear`'s f64 grad custom autograd
path. The candidate replaced owned `ctx.save_for_backward(x/w)` copies with
`tensor_apply_function_f64_borrowed_inputs`, so backward borrowed immutable
input and weight slices from the tape. Forward math, backward math, bias
handling, input order, gradient order, shape handling, dtype, and RNG were
unchanged.

## Baseline

Initial baseline:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- linear_train/hidden/2048 --warm-up-time 1 --measurement-time 5 --sample-size 20
worker selected: vmi1149989
linear_train/hidden/2048 time: [99.973 ms 108.90 ms 119.26 ms]
```

RCH did not keep the after-run on `vmi1149989`, so this initial run was not used
for the keep/reject gate.

Matched same-worker baseline:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-api --bench ops_bench -- linear_train/hidden/2048 --warm-up-time 1 --measurement-time 5 --sample-size 20
worker selected: ts1
linear_train/hidden/2048 time: [60.221 ms 61.203 ms 62.513 ms]
```

## Candidate

Cross-worker signal run, not used for gate:

```text
worker selected: vmi1293453
linear_train/hidden/2048 time: [48.942 ms 50.233 ms 51.613 ms]
```

Matched same-worker candidate:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-api --bench ops_bench -- linear_train/hidden/2048 --warm-up-time 1 --measurement-time 5 --sample-size 20
worker selected: ts1
linear_train/hidden/2048 time: [56.227 ms 57.677 ms 59.477 ms]
```

Same-worker median ratio: `61.203 / 57.677 = 1.061x`.

Score: `Impact 1.06 x Confidence 0.95 / Effort 1.0 = 1.01`.

## Proof

Focused candidate proof before source removal:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-api functional_linear -- --nocapture
worker selected: vmi1293453
result: 5 passed; 0 failed; 1921 filtered out
```

The first test run exposed a candidate-local unused `ws` warning. The candidate
was updated to `_ws`, then the focused tests passed again. Remaining warnings
were pre-existing `unused_mut` warnings in unrelated test closures.

Golden-output verification after rejection:

```text
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
result: all present golden outputs OK
```

UBS note: `ubs crates/ft-api/src/lib.rs` hung inside an `ast-grep` substep on
its shadow copy and was terminated after it stopped making observable progress.

## Isomorphism Ledger

- Ordering/tie-breaking: no sorting, maps, ties, or iteration-order changes.
  Custom-op input order remained `[input, weight, bias?]`; gradient order
  remained `[dx, dw, db?]`.
- Floating point: forward stayed in `linear_tensor_f64`; backward stayed in
  `linear_backward_f64`; no loop order, accumulation order, GEMM dimensions, or
  bias-gradient formula changed.
- RNG: the benchmark's random inputs are constructed before `functional_linear`;
  the candidate added no RNG calls and removed none.
- Mutation: the candidate relied on the existing borrowed-input custom-op
  contract. Because the same-worker impact was below gate, no additional
  mutation/version contract tests were kept.

## Verdict

Rejected. The lever is behavior-isomorphic for the focused path and produces a
small win, but the measured `1.061x` same-worker gain scores below the required
`Score >= 2.0` keep threshold. The source hunk was removed.

Next route: stop micro-tuning `save_for_backward` overhead for Linear and
re-profile for a materially deeper primitive, such as a fused training-step
Linear/GEMM backward workspace plan, a batched parameter-gradient accumulation
layout, or a different ready `[perf]` bead with a larger measured residual.
