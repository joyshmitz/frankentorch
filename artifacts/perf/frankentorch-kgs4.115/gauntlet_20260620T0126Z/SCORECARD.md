# frankentorch-kgs4.115 Scorecard

Agent: `IvoryDeer`

## Measurements

| Check | Command | Result |
|---|---|---|
| Current Rust A/B | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo run --release -p ft-api --example group_norm_f32_grad_ab` | rch `hz1`; composed `101.96 ms`, fused current `11.72 ms`, speedup `8.70x` |
| Parent baseline | `RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a-baseline rch exec -- cargo run --release -p ft-api --example group_norm_f32_grad_ab` from detached parent `e1927d48` | rch `hz1`; composed `165.03 ms`, fused parent `19.13 ms`; current is `1.63x` faster |
| PyTorch oracle | local CPU torch f32 GroupNorm best-of-12, matching `[8,64,28,28]`, groups `32`, affine grads, sum loss | PyTorch `2.12.1+cpu`; best `0.615446 ms`, median `0.989997 ms`; current FrankenTorch best is `19.04x` slower |

## Validation

| Gate | Command | Result |
|---|---|---|
| Kernel bit guard | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo test -p ft-kernel-cpu group_norm_f32_unit_dy_matches_general_reference_bits -- --nocapture` | passed on `vmi1153651`: 1 passed, 0 failed |
| API gradient parity | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo test -p ft-api functional_group_norm_f32_grad_matches_f64_path -- --nocapture` | passed on `ovh-a`: 1 passed, 0 failed |
| Conformance | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo test -p ft-conformance strict_scheduler -- --nocapture` | passed on `vmi1152480`: `strict_scheduler_conformance_is_green` passed |

## Release Decision

- Internal verdict: keep. Same-worker current-vs-parent fused f32 path improved `19.13 ms -> 11.72 ms`.
- PyTorch verdict: loss. Current remains `19.04x` slower than CPU PyTorch on the same local oracle workload.
- Score: `0W / 1L / 0N` versus PyTorch.
- Follow-up route: no more tiny `dy == 1` GroupNorm branches without a fresh profile; move to allocation/tape/fusion/parallel scheduling.
