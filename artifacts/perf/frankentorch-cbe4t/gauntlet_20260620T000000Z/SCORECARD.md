# frankentorch-cbe4t Scorecard

## Lever

Delay `TensorTape::backward_with_options` gradient buffer materialization until the
first contribution reaches a node. Each slot records the expected gradient length,
so shape mismatches remain fail-closed. First-contribution initialization uses
`0.0 + contribution` to preserve the old eager-zero-buffer f64 arithmetic, including
signed-zero behavior, while fan-in keeps the existing `+=` path.

Source ideas: alien-graveyard region/arena reset guidance, allocator hot-path evidence
from `frankentorch-96e5d`, and profile-first allocation-elision discipline from the
gauntlet/performance skills.

## Verdict

KEEP.

This is a real local PyTorch-enabled gap reduction on the `avg_pool1d` train row whose
root cause was already proven to be generic backward allocation/first-touch overhead.
It is not a PyTorch victory; FrankenTorch still loses this workload.

## Head-to-head

Command:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a \
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
cargo bench -p ft-api --bench pytorch_gauntlet_bench -- avg_pool1d --noplot
```

| Case | Baseline median | Candidate median | Result |
| --- | ---: | ---: | --- |
| FrankenTorch `gauntlet_avg_pool1d_grad/frankentorch_kgs4_122` | 89.360 ms | 70.206 ms | WIN, 1.27x faster / -21.4% |
| PyTorch `pytorch_2_12_cpu` | 6.7081 ms | 6.9328 ms | NEUTRAL/noisy |
| FT / PyTorch ratio | 13.32x slower | 10.13x slower | LOSS remains; gap reduced 1.31x |

PyTorch W/L/N for this row: `0 / 1 / 0`.

## Remote RCH Evidence

Command:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a \
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
rch exec -- cargo bench -p ft-api --bench pytorch_gauntlet_bench -- avg_pool1d --noplot
```

| Worker | Baseline median | Candidate median | Status |
| --- | ---: | ---: | --- |
| `ovh-a` | 73.254 ms | 69.674 ms | FT-only neutral, p=0.17 |
| `hz2` | n/a | 101.92 ms | routed candidate only |

Both remote PyTorch arms failed with `ModuleNotFoundError: No module named 'torch'`.
The `hz2` row is routing/environment evidence only because the worker differed from
the baseline worker and no PyTorch arm ran.

## Validation

All build/test commands used `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a`.

| Gate | Command | Result |
| --- | --- | --- |
| Compile | `rch exec -- cargo check -p ft-autograd --all-targets` | PASS |
| Autograd tests | `rch exec -- cargo test -p ft-autograd --lib` | PASS, 476/0 |
| API avg_pool1d bit regression | `rch exec -- cargo test -p ft-api functional_avg_pool1d_fused_matches_reshape_2d_forward_and_backward_bits --lib` | PASS, 1/0 |
| Strict conformance | `rch exec -- cargo test -p ft-conformance strict_scheduler_conformance_is_green --lib` | PASS, 1/0 |
| Clippy | `rch exec -- cargo clippy -p ft-autograd --all-targets -- -D warnings` | PASS |
| Whitespace | `git diff --check` | PASS |
| Format | `cargo fmt --check`; `cargo fmt -p ft-autograd --check`; `rustfmt --edition 2024 --check crates/ft-autograd/src/lib.rs` | FAILS on pre-existing unrelated formatting drift outside this hunk; formatter not run |
| UBS | `ubs crates/ft-autograd/src/lib.rs` | COMPLETED; reports the existing whole-file inventory, including pre-existing panic/unwrap/token-comparison heuristics outside this hunk |

## Follow-up Routing

The user asked the next pass to target matmul/conv PyTorch gaps. Do that next; do not
spend another pass on generic avg_pool1d kernel paths unless a fresh profile moves the
hotspot away from backward allocation.
