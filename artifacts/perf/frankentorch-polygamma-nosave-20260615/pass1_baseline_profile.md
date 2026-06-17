# frankentorch-polygamma-nosave-20260615 Pass 1 Baseline/Profile Contract

## Command

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -j 1 -p ft-api --bench special_bench -- polygamma2_1m --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

## Worker

- RCH selected worker: `ovh-a` at `ubuntu@51.222.245.56`.
- RCH transcript: `artifacts/perf/frankentorch-polygamma-nosave-20260615/baseline_polygamma2_1m_rch.log`.

## Baseline

- Criterion row: `polygamma2_1m`.
- Interval/median: `[21.503 ms 21.835 ms 22.150 ms]`.
- Result is crate-scoped to `ft-api` and uses the requested `special_bench` row only.

## Hotspot Rationale

- The benchmark constructs a no-grad `tensor_rand(vec![1_000_000], false)` input and calls `tensor_polygamma(2, x)`.
- `tensor_polygamma` routes through `tensor_apply_function`; the forward closure unconditionally runs `ctx.save_for_backward(vals.to_vec(), shape.to_vec())` before computing `par_map_f64(vals, |x| polygamma_approx(n, x))`.
- Since this row uses `requires_grad=false`, any saved input context is dead unless the runtime needs it for a grad-enabled path. The suspected one-lever target is to avoid the no-grad saved `vals.to_vec()` work without changing the scalar `polygamma_approx` arithmetic.

## Next-Pass Isomorphism Obligations

- Preserve `polygamma_approx(n, x)` evaluation order and floating-point behavior for every element.
- Preserve grad-mode behavior exactly: saved tensor contents, saved shape, backward `polygamma(n + 1, x)`, gradient length checks, and error behavior.
- Preserve no-grad observable tensor metadata, dtype, shape, storage contents, tensor registration, DAC/evidence ledger behavior, and strict-mode semantics.
- Keep the lever isolated to no-grad saved-context elimination; do not combine it with small-order numeric rewrites or recurrence/asymptotic `powf` changes.
- Prove behavior before acceptance with focused `ft-api` tests and compare any candidate benchmark on the same worker when possible.
