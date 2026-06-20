# frankentorch-kgs4.119 scorecard

Date: 2026-06-20
Status: KEEP existing borrowed-input Conv3d autograd path; PyTorch loss remains.

| Gate | Command | Result |
| --- | --- | --- |
| Compile | `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench` | Passed on `vmi1152480`. |
| PyTorch gauntlet | `PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b cargo bench -p ft-api --bench pytorch_gauntlet_bench -- 'gauntlet_conv3d_grad' --noplot` | Local current FrankenTorch median `24.095 ms`; PyTorch `2.12.1+cpu` median `10.126 ms`; FrankenTorch is `2.38x` slower. |
| Same-worker A/B | `rch exec -- cargo bench -p ft-api --bench pytorch_gauntlet_bench -- 'gauntlet_conv3d_grad/frankentorch_kgs4_119' --noplot` | On `ovh-a`, disabled save-copy median `19.429 ms`; current borrowed-input median `15.632 ms`; current is `1.24x` faster. |
| Routing-only remote row | same command | Initial current-only run on `vmi1152480` median `28.364 ms`; no disabled/PyTorch comparator on that worker, so not used for keep/reject. |
| Conv3d API tests | `rch exec -- cargo test -p ft-api conv3d --lib -- --nocapture` | Passed on `vmi1293453`: `10 passed; 0 failed`. |
| Conformance | `rch exec -- cargo test -p ft-conformance` | Passed on `vmi1227854`: lib `199/0`, bins/integration/smoke/doc tests all green. |
| Clippy | `rch exec -- cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings` | Passed on `vmi1227854`. |
| Formatting | `rustfmt --edition 2024 --check crates/ft-api/benches/pytorch_gauntlet_bench.rs`; `python3 -m py_compile crates/ft-api/benches/pytorch_conv3d_grad.py`; `git diff --check` | All passed. |

Verdict: the borrowed-input path should remain. The remaining PyTorch gap should
route to whole training-row overhead rather than restoring saved input copies.
