# frankentorch-kgs4.120 - RMSNorm unit-dy backward fused path

## Claim

Measured no-ship. The f64 all-ones-`dy` RMSNorm backward branch did not beat
the generic path in same-worker release-profile Criterion, and the final
generic source remains slower than PyTorch.

## Profile target

- Current realistic train reprofile includes `rms_norm/grad_2048x1024` around 150 ms in `artifacts/perf/frankentorch-next-reprofile-20260617/current_top_train_reprofile.log`.
- Later reprofiles still keep RMSNorm grad in the active top set:
  - `artifacts/perf/frankentorch-next-reprofile-20260617b/current_top_train_reprofile_after_6olvt.log`
  - `artifacts/perf/frankentorch-next-reprofile-20260617c/current_top_train_reprofile_after_16m8a.log`
- The Criterion workload is `loss = tensor_sum(functional_rms_norm(...))`, so the upstream gradient entering RMSNorm backward is exactly all `+1.0`.

## Lever

The rejected lever detected finite exact all-ones `dy` in
`rms_norm_backward_f64` and routed it to a fused unit-dy path that:

- removes `dy` loads and multiplies from the `dx` and `dweight` loops,
- computes per-row `rstd` once inside backward and reuses it for `dweight`,
- keeps the old generic formula for non-ones, NaN, or infinite inputs.

This is a cache/memory-traffic lever from the no-gaps campaign: avoid streaming a 2M-element all-ones gradient tensor through every reduction when the training trace already proves it is a constant.

## Negative-evidence ledger

- `frankentorch-fad7c` rejected forward-saved RMSNorm rstd reuse. Baseline on worker `vmi1227854`: `[117.17 ms, 118.95 ms, 120.76 ms]`; candidate was only overlapping or regressed, with local supplemental `[120.04 ms, 123.04 ms, 126.15 ms]`. Do not retry forward saved-stats sidecars.
- This attempt was not a forward sidecar. It specialized the sum-loss backward
  dataflow and preserved the generic formula for every non-finite or
  non-unit-dy case.
- Same-worker release-profile Criterion on `vmi1153651`:
  - active branch: `[51.215 ms, 59.289 ms, 67.477 ms]`
  - generic-disabled probe: `[52.546 ms, 58.407 ms, 64.377 ms]`,
    `p=0.55`, no detected change
  - final branch-removed source: `[46.294 ms, 64.615 ms, 87.183 ms]`,
    `p=0.58`, no detected change
- The active branch was `1.0151x` slower than the generic-disabled median.
  It was removed.
- Local PyTorch CPU `2.12.1+cpu`, 32 threads, same f64 shape and scalar loss,
  measured `13.241798 ms` median. The final FrankenTorch source is `4.8796x`
  slower by this mixed-location ratio.
- Do not retry this f64 unit-dy branch family. A retry must move below this
  abstraction boundary: automatic scalar-loss fusion in the tape/session,
  persistent row-stat/workspace reuse, arena/bump allocation for graph and grad
  buffers, f64-native layout work, or generated fused RMSNorm-sum code with a
  same-worker keep gate.

## Correctness guard

The now-misleading branch-specific bit-reference guard was removed with the
branch. The generic f64 RMSNorm backward remains covered by full
`ft-kernel-cpu` library tests, API RMSNorm gradient tests, and strict scheduler
conformance.

## Verification

- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo bench -p ft-api --bench ops_bench --profile release -- rms_norm/grad_2048x1024 --sample-size 10 --warm-up-time 1 --measurement-time 3 --noplot`
- `rch exec -- cargo test -p ft-kernel-cpu --lib -- --nocapture`: passed,
  `504 passed; 0 failed; 2 ignored`
- `rch exec -- cargo test -p ft-api functional_rms_norm --lib -- --nocapture`:
  passed, `6 passed; 0 failed`
- `rch exec -- cargo test -p ft-conformance strict_scheduler -- --nocapture`:
  passed, strict-scheduler conformance green
- `rch exec -- cargo check -p ft-kernel-cpu --lib`: passed
- `rch exec -- cargo clippy -p ft-kernel-cpu --lib -- -D warnings`: passed
- `rustfmt --edition 2024 --check crates/ft-kernel-cpu/src/lib.rs` remains
  blocked by existing whole-file drift outside this lane; no broad reformat was
  applied.
