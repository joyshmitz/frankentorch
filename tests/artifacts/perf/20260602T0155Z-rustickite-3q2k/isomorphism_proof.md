# frankentorch-3q2k Isomorphism Proof

## Lever

`functional_linear` now passes a 1-D bias directly to `tensor_addmm` instead of
first materializing a full `[batch, out_features]` expanded bias tensor.

## Preserved Semantics

- Output ordering: unchanged. `addmm` still writes row-major `[m, n]` output.
- Tie-breaking: not applicable. No comparisons or ordering decisions changed.
- Floating-point behavior: unchanged for GEMM and final `beta * bias + alpha * g`
  arithmetic. The previous expanded-bias path copied the same bias value into
  each row before `addmm`; the new path lets `addmm` select the same column bias
  with `col = i % n`.
- RNG behavior: unchanged. No random state is read or advanced.
- Diagnostics and shape behavior: unchanged for the optimized benchmark path.
  Non-2D matrix inputs still fail closed in `addmm`.
- Gradient behavior: unchanged. The previous path accumulated bias gradients via
  expand backward; `addmm`'s 1-D input backward sums `grad_out` along rows and
  scales by `beta`, which is the same reduction for this call (`beta = 1.0`).

## Golden Output

`golden_linear_bias_outputs.txt` records a deterministic batched linear case with
1-D bias and the resulting bias gradient. The sha256 is recorded in
`golden_checksums.txt`.

## Benchmark

- Baseline: same-worker rch Criterion on `ts2`, `linear_forward/hidden/2048`
  p50 `21.520 ms`.
- After: same-worker rch Criterion on `ts2`, `linear_forward/hidden/2048`
  p50 `20.969 ms`.
- Delta: `0.551 ms` faster, `2.56%` faster.
- Score: impact 1 x confidence 2 / effort 1 = 2.0.

## Validation

- `rch exec -- cargo bench -p ft-api --bench ops_bench -- linear_forward/hidden/2048 --warm-up-time 1 --measurement-time 5 --sample-size 20`: p50 `20.969 ms` after same-worker baseline p50 `21.520 ms`.
- `sha256sum -c tests/artifacts/perf/20260602T0155Z-rustickite-3q2k/golden_checksums.txt`: passed.
- `git diff --check -- crates/ft-api/src/lib.rs .skill-loop-progress.md tests/artifacts/perf/20260602T0155Z-rustickite-3q2k`: passed.
- `rch exec -- cargo test -p ft-api functional_linear`: passed.
- `rch exec -- cargo check -p ft-api --all-targets`: passed with existing warnings.
- `rch exec -- cargo clippy -p ft-api --all-targets -- -D warnings`: blocked by the existing ft-api lint backlog.
- `rch exec -- cargo fmt -p ft-api --check`: blocked by existing broad ft-api/ops_bench formatting drift.
- `ubs crates/ft-api/src/lib.rs`: blocked by existing file-wide scanner backlog; report counted 311 critical, 16115 warning, and 1719 info findings, with no unsafe blocks detected.
