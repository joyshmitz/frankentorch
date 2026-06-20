# frankentorch-kgs4.120 - f64 RMSNorm unit-dy no-ship

## Verdict

Reject and remove the f64 `rms_norm_backward_f64` all-ones-`dy` branch. The
branch was a radical but too-local memory-traffic specialization: detect scalar
loss upstream, skip dense `dy` loads, precompute row `rstd`, and reuse it for
`dweight`. Same-worker Criterion did not show a keepable gain, and the final
generic path remains far behind PyTorch.

## Workload

- FrankenTorch benchmark: `rms_norm/grad_2048x1024`
- Shape: f64 `[2048,1024]`, affine weight, scalar-sum loss
- Command form: `rch exec -- cargo bench -p ft-api --bench ops_bench --profile release -- rms_norm/grad_2048x1024 --sample-size 10 --warm-up-time 1 --measurement-time 3 --noplot`
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b`
- Same-worker proof worker: `vmi1153651`

## Results

| Source | Median | Interval | Notes |
|---|---:|---:|---|
| Active f64 unit-dy branch | `59.289 ms` | `[51.215 ms, 67.477 ms]` | candidate branch |
| Generic-disabled probe | `58.407 ms` | `[52.546 ms, 64.377 ms]` | `p=0.55`, no detected change |
| Final branch-removed source | `64.615 ms` | `[46.294 ms, 87.183 ms]` | `p=0.58`, no detected change |
| PyTorch CPU `2.12.1+cpu` | `13.241798 ms` | min `6.298722 ms`, p95 `17.442162 ms` | local mixed-location comparator |

Ratios:

- Active branch / generic-disabled: `1.0151x` slower.
- Active branch / PyTorch: `4.4774x` slower.
- Generic-disabled / PyTorch: `4.4110x` slower.
- Final branch-removed source / PyTorch: `4.8796x` slower.

Win/loss/neutral vs PyTorch: `0W / 1L / 0N`.

## Source Changes

Removed:

- f64 all-ones-`dy` branch in `rms_norm_backward_f64`
- `rms_norm_backward_f64_unit_dy_finite`
- branch-specific bit-reference test

The generic f64 RMSNorm backward remains the only product path.

## Evidence

- `current_active_rch_rms_norm_grad.log`
- `generic_disabled_rch_rms_norm_grad.log`
- `final_removed_rch_rms_norm_grad.log`
- `local_pytorch_rms_norm_f64_sum.log`
- `test_ft_kernel_cpu_lib.log`
- `test_ft_api_functional_rms_norm.log`
- `test_ft_conformance_strict_scheduler.log`
- `check_ft_kernel_cpu_lib.log`
- `clippy_ft_kernel_cpu_lib.log`
- `rustfmt_ft_kernel_cpu_check.log`
- `git_diff_check.log`
- `ubs_scoped.log`

## Gates

- `rch exec -- cargo test -p ft-kernel-cpu --lib -- --nocapture`: passed,
  `504 passed; 0 failed; 2 ignored`.
- `rch exec -- cargo test -p ft-api functional_rms_norm --lib -- --nocapture`:
  passed, `6 passed; 0 failed`.
- `rch exec -- cargo test -p ft-conformance strict_scheduler -- --nocapture`:
  passed, strict scheduler conformance green.
- `rch exec -- cargo check -p ft-kernel-cpu --lib`: passed.
- `rch exec -- cargo clippy -p ft-kernel-cpu --lib -- -D warnings`: passed.
- `git diff --check`: passed.
- `ubs` on the scoped source/docs/artifact summary surface: passed with `0`
  critical issues; it still reports the existing broad warning inventory in
  the large kernel file.
- `rustfmt --edition 2024 --check crates/ft-kernel-cpu/src/lib.rs`: blocked by
  existing whole-file drift outside this lane; no broad reformat was applied.

## Retry Boundary

Do not retry a local f64 unit-dy branch that guard-scans `dy`, `x`, and
`weight` and materializes per-row `rstds` inside backward. The next RMSNorm
attempt must move below this abstraction boundary: automatic scalar-loss
fusion, persistent row-stat/workspace reuse, tape/session arena allocation,
f64-native layout, or generated fused RMSNorm-sum code with a same-worker keep
gate.
