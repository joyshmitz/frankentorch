# frankentorch-kgs4.137 RMSNorm Scalar-Sum Scorecard

## Verdict

Rejected and reverted from product source. The scalar-sum RMSNorm path did not
beat the existing materialized path on the same RCH worker, so no source keep.

## Workload

- Bench: `ops_bench` `rms_norm/grad_2048x1024`
- Shape: f64 `[2048,1024]`
- Loss: scalar sum over RMSNorm output
- Target dir: `/data/projects/.rch-targets/frankentorch-cod-b`
- Run dir: `artifacts/perf/frankentorch-kgs4.137/gauntlet_20260620T133307Z`

## Same-Worker FT Gate

| Arm | Worker | Median | Interval |
| --- | --- | ---: | --- |
| Materialized baseline, isolated | `vmi1227854` | `12.229 ms` | `[11.683, 12.596]` |
| Materialized baseline, candidate run | `vmi1227854` | `12.086 ms` | `[11.334, 13.179]` |
| Scalar-sum candidate | `vmi1227854` | `12.329 ms` | `[11.023, 13.944]` |

Candidate/materialized ratio: `12.329 / 12.086 = 1.020x` latency. That is
about `2.0%` slower and inside Criterion noise, so the keep gate fails.

## PyTorch Ratio

Local PyTorch oracle: torch `2.12.1+cpu`, 32 threads, same shape/dtype, 30 reps.

| Metric | PyTorch | Candidate FT/PyTorch |
| --- | ---: | ---: |
| min | `4.994618 ms` | `2.47x` slower |
| median | `14.360424 ms` | `0.858x` latency |
| mean | `13.693821 ms` | `0.900x` latency |
| p95 | `19.172968 ms` | `0.643x` latency |

The median comparison is FT-favorable, but it is advisory only. The decisive
same-worker FT A/B showed no internal improvement.

## Product Decision Ratio

Win/loss/neutral vs PyTorch for this product decision: `0W / 0L / 1N`.

## Gates

- Symbol scrub: no remaining `functional_rms_norm_sum`,
  `rms_norm_sum_forward`, `rms_norm_backward_scalar`, or `grad_sum_2048x1024`
  references in the touched API/kernel/bench source.
- Post-revert bench: `cargo bench -p ft-api --bench ops_bench --
  rms_norm/grad_2048x1024 --warm-up-time 1 --measurement-time 3
  --sample-size 10 --noplot` passed on RCH `hz2`, median `29.419 ms`.
- `cargo test -p ft-kernel-cpu rms_norm --lib -- --nocapture`: passed, 2
  focused tests.
- `cargo test -p ft-api functional_rms_norm --lib -- --nocapture`: passed, 6
  focused tests.
- `cargo test -p ft-conformance`: passed. RCH had no admissible workers and
  fell back local, but the per-crate conformance suite completed green.
- `cargo clippy -p ft-api --lib -- -D warnings`: passed after a current-checkout
  BatchNorm scalar-sum range loop was rewritten to iterate over `x`.
- `cargo clippy -p ft-kernel-cpu --lib -- -D warnings`: passed.
- `cargo fmt --check -p ft-api -p ft-kernel-cpu`: passed.
- `git diff --check` on the touched docs/kernel/artifact surface: passed.
- `ubs` on the touched docs/kernel/artifact surface: `0` critical issues;
  broad existing warning inventory remains.

## Retry Rule

Do not retry another RMSNorm scalar-loss wrapper that only removes `tensor_sum`
or a dense constant `dy`. The next RMSNorm attempt needs a deeper tape,
workspace, or row primitive and same-worker proof.
