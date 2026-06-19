# FrankenTorch Release-Readiness Scorecard

Updated: 2026-06-19

## Performance Gauntlet

| Bead | Workload | Result vs PyTorch | Before/after verdict | Release action |
|---|---:|---:|---:|---|
| `frankentorch-kgs4.124` | SmoothL1 f64 mean-loss backward, 8M elems | `1.99x` slower | internal keep; `963.16 ms` -> `757.63 ms` on `hz2` | kept; follow-up `frankentorch-kgs4.128` |
| `frankentorch-kgs4.126` | max_pool1d f64 train step `[8,64,8192]` | `12.31x` slower | no gain; candidate median `184.41 ms` vs parent `178.47 ms` | reverted |

Score: `2/5` for the measured gauntlet lanes. Correctness guards are green and
the SmoothL1 scalar reduced-grad lever is a real internal speedup, but neither
measured workload is performance-dominant against PyTorch yet.

## Current Gates

| Gate | Scope | Result |
|---|---|---|
| Criterion | `cargo bench -p ft-api --bench ops_bench -- smooth_l1/grad_8m --noplot` | completed on `hz2`; current median `757.63 ms` |
| PyTorch oracle | `torch_smooth_l1_grad_8m.py` | local PyTorch `2.12.1+cu130` median `373.61 ms` |
| Compile | `rch exec -- cargo check -p ft-api` | passed |
| Correctness | `rch exec -- cargo test -p ft-kernel-cpu smooth_l1_backward_reduced_f64_matches_uniform_dloss_bits` | passed |
| Correctness | `rch exec -- cargo test -p ft-api smooth_l1_loss_reduced_grad_matches_materialized_reference_bits` | passed after explicit default-argument test-call fix |
| Criterion | `cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool1d --noplot` | completed locally with PyTorch `2.12.1+cpu` |
| Compile | `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench` | passed on `ovh-a` for final harness |
| Correctness | `rch exec -- cargo test -p ft-kernel-cpu max_pool1d_direct_matches_2d_h1_first_tie_forward_backward_bit_exact` | passed on `ovh-a` |
| Formatting | `rustfmt --edition 2024 --check crates/ft-api/benches/pytorch_gauntlet_bench.rs` | passed |

Known caveat: `cargo fmt --check -p ft-api` remains blocked by pre-existing
crate-wide formatting debt in unrelated examples and long `ft-api/src/lib.rs`
regions. `cargo fmt -p ft-api -- --check` also reports broad existing drift
for the SmoothL1 closeout; follow-up `frankentorch-6xsy8` tracks that cleanup.

UBS caveat: a full changed-file UBS scan including the 136k-line
`ft-api/src/lib.rs` did not complete after several minutes and was interrupted
or timed out in the pre-commit hook. The max_pool1d benchmark surface was
scanned directly and passed with zero critical or warning findings; no UBS
verdict was available for the SmoothL1 `ft-api/src/lib.rs` closeout.

## Next Perf Target

The `.124` result points toward deeper SmoothL1 training overhead: tape setup,
input/materialization cost, loss backward kernel shape, SIMD, and cache layout.
The `.126` result points away from tiny max_pool1d backward scatter branches and
toward larger full-step costs: autograd/session setup, allocation churn, and
forward saved-index materialization. Future work should profile those frames
before trying another scalar-wrapper or one-off unit-gradient branch.
