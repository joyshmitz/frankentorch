# frankentorch-kgs4.133 Scorecard

## Workload

- FrankenTorch: `ops_bench` `conv2d/grad_hw/64`
- Shape: f64 input `[4,64,64,64]`, f64 weight `[64,64,3,3]`
- Operation: Conv2d stride 1, padding 1, scalar `sum` loss, backward
- PyTorch oracle: local CPU PyTorch `2.12.1+cpu`, 32 compute threads, 32 interop threads

## Candidate

Activate the parked `conv2d_backward_f64` all-ones-`dout` specialization:
compute one shared `dweight` row and one shared `dpanel` row, then broadcast the
row or scatter it through the existing col2im geometry.

## Result

| Arm | Evidence | Median |
|---|---|---:|
| Current FrankenTorch | `baseline_current_rch_ops_conv2d_grad_hw64.log` on `vmi1152480` | `121.07 ms` |
| Active candidate | `candidate_active_rch_ops_conv2d_grad_hw64.log` on `vmi1152480` | `117.92 ms` |
| PyTorch CPU | `local_pytorch_conv2d_f64_grad_hw64.log` | `63.449849 ms` |

Criterion reported `[-7.9705% -2.5970% +2.8489%]`, `p = 0.38`, and
`No change in performance detected`.

## Verdict

Rejected. The active candidate stayed within noise and remained `1.86x` slower
than PyTorch. Product source removes the compile-time-false parked branch.

## Final Gates

- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo check -p ft-kernel-cpu --all-targets`: passed on `hz1`; existing example warning in `gemm_golden.rs` remains outside this branch removal.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo clippy -p ft-kernel-cpu --lib -- -D warnings`: passed on `vmi1293453`.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`: blocked by pre-existing lint debt in examples/tests and unrelated helper code, not by the removed conv2d branch.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo test -p ft-kernel-cpu conv2d -- --nocapture`: passed, 5 tests ok and 1 perf-only ignored.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo test -p ft-conformance`: passed on `vmi1227854`, including 199 lib tests, bin unit tests, integration tests, smoke tests, and doc-tests.
- `git diff --check`: passed.
- `rustfmt --edition 2024 --check crates/ft-kernel-cpu/src/lib.rs`: blocked by pre-existing whole-file formatting drift outside this deletion.
- `ubs crates/ft-kernel-cpu/src/lib.rs docs/NEGATIVE_EVIDENCE.md docs/RELEASE_READINESS_SCORECARD.md artifacts/perf/frankentorch-kgs4.133/gauntlet_20260620T0533Z/SCORECARD.md artifacts/perf/frankentorch-kgs4.133/gauntlet_20260620T0533Z/NEGATIVE_EVIDENCE_LEDGER.md`: 0 critical issues; existing large-file warning inventory remains.

## Follow-Up

Do not retry this exact materialized-im2col all-ones row-collapse shape. Next
conv2d work should require fresh profile evidence for a different primitive:
direct no-panel all-ones backward, workspace-backed panel reuse, cache-blocked
col2im, arena-backed temporary storage, f32-native end-to-end ratio work, or a
fused loss/backward path.
