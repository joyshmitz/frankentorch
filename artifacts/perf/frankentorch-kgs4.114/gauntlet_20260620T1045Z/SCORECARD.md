# frankentorch-kgs4.114 Scorecard

## Decision

Rejected and reverted. The f32 BatchNorm all-ones-`dy` branch made the
same-worker rch row slower on `vmi1152480`.

## Measurements

| Run | Machine | Row | Median | Ratio |
|---|---|---|---:|---:|
| Active branch | local | FrankenTorch | `228.85 ms` | `33.29x` slower than PyTorch |
| Active branch | local | PyTorch 2.12 CPU | `6.8744 ms` | baseline oracle |
| Disabled/final | local | FrankenTorch | `238.33 ms` | `28.14x` slower than PyTorch |
| Disabled/final | local | PyTorch 2.12 CPU | `8.4699 ms` | baseline oracle |
| Disabled/final | rch `vmi1152480` | FrankenTorch | `147.30 ms` | baseline for A/B |
| Active branch | rch `vmi1152480` | FrankenTorch | `157.93 ms` | `1.072x` slower than disabled |

`current_local_ft_only_batch_norm2d_f32_rerun.log` is retained as a
non-evidence artifact: the filtered local rerun returned without Criterion
benchmark rows.

## Gates

| Gate | Result |
|---|---|
| `rch exec -- cargo test -p ft-kernel-cpu batch_norm_f32_unit_dy_matches_general_reference_bits -- --nocapture` | passed on `hz2` |
| `rch exec -- cargo test -p ft-api functional_batch_norm2d_f32_grad_matches_f64_path -- --nocapture` | passed via rch local fallback |
| `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench` | passed on `vmi1156319` |
| `rch exec -- cargo clippy -p ft-kernel-cpu --lib -- -D warnings` | passed on `vmi1293453` |
| `rch exec -- cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings` | passed on `vmi1153651` |
| `rch exec -- cargo test -p ft-conformance` | passed on `hz2` |
| `rustfmt --edition 2024 --check crates/ft-api/benches/pytorch_gauntlet_bench.rs` | passed |
| `python3 -m py_compile crates/ft-api/benches/pytorch_batch_norm2d_f32_grad.py` | passed |
| `git diff --check` on touched files | passed |

## Route

The remaining BatchNorm gap is not in this narrow `dy == 1` branch. Route any
next attempt to whole-row work: fused scalar-loss BatchNorm, saved-stat reuse,
persistent workspaces, arena-backed tape/tensor allocation, stats+backward
fusion, or a cache-blocked f32 reduction plan with same-worker proof.
