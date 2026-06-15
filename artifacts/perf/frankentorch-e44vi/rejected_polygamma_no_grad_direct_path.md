# frankentorch-e44vi rejection: polygamma no-grad direct path

## Target

- Bead: `frankentorch-e44vi`
- Target row: `ft-api` Criterion `special_bench/polygamma2_1m`
- Profile-backed baseline source:
  `artifacts/perf/frankentorch-special-reprofile-20260615/baseline_special_bench.log`
- Hot row on `ovh-a`: `polygamma2_1m [24.150 ms 24.524 ms 25.009 ms]`

## Candidate

One bit-preserving lever was attempted after rejecting the more aggressive
small-order multiplication-chain idea as FP-bit risky:

- keep `polygamma_approx` scalar arithmetic unchanged
- keep the grad-mode custom autograd path unchanged
- add a no-grad direct storage path for `tensor_polygamma`
- avoid `tensor_apply_function` context/saved-tensor cloning for no-grad inference

The first candidate compile failed because `tensor_variable_from_storage` returns
`TensorNodeId`, not `Result<TensorNodeId>`. The retry fixed that mechanical issue.

## Evidence

Fresh baseline on `vmi1153651` was recorded but is not used for acceptance because
the candidate selected a different worker:

- `baseline_polygamma2_criterion.log`
- `polygamma2_1m [60.486 ms 77.837 ms 98.012 ms]`

Same-worker acceptance comparison uses `ovh-a`:

- Baseline: `frankentorch-special-reprofile-20260615/baseline_special_bench.log`
- `polygamma2_1m [24.150 ms 24.524 ms 25.009 ms]`
- Candidate retry: `candidate_polygamma2_criterion_retry.log`
- `polygamma2_1m [28.475 ms 29.000 ms 29.283 ms]`

Result: regression, `24.524 / 29.000 = 0.85x`.

## Verdict

Reject. Score `0.0`.

The source hunk was removed. No `ft-api` source diff remains for this bead.

## Reroute

Do not repeat clone/context bypasses for this row. The time is in the scalar
`polygamma_approx` recurrence/asymptotic `powf` work. The safe next route is an
algorithmically different numeric primitive that still satisfies FP parity:

- produce a shadow implementation for orders `1/2/3`
- prove exact output policy against the current golden SHA before dispatch
- only wire it if bit policy is explicitly preserved or the compatibility mode
  accepts a reference-improving numeric change
