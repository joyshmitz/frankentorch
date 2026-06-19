# frankentorch-kgs4.131 Scorecard

## Lever

Move completed `TensorTape::backward_with_options` gradient buffers into
`TensorBackwardReport` instead of cloning every reachable `Vec<f64>` after the
tape walk.

Rationale source: alien-graveyard region/arena ownership and reverse-mode AD
tape guidance; extreme-software-optimization profile-first clone elimination.

## Keep/Reject

KEEP.

This is a measured backward/report-materialization win with no correctness
regression in the focused checks.

## Same-worker FT A/B

Command:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
rch exec -- cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool3d --noplot
```

Worker: `hz1` for both baseline and candidate.

| Case | Baseline median | Candidate median | Result |
| --- | ---: | ---: | --- |
| `gauntlet_max_pool3d_grad/frankentorch_kgs4_117` | 38.283 ms | 30.620 ms | WIN, 1.25x faster / -20.0%, p=0.00 |

Remote PyTorch arm failed in both rch runs because the worker lacks the `torch`
module. That failure is recorded in `NEGATIVE_EVIDENCE_LEDGER.md`.

## Local PyTorch-enabled Head-to-head

Command:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool3d --noplot
```

| Case | Baseline median | Candidate median | Result |
| --- | ---: | ---: | --- |
| FrankenTorch headline | 15.718 ms | 6.6743 ms | WIN, 2.36x faster / -57.5%, p=0.00 |
| PyTorch headline | 1.7717 ms | 1.6898 ms | NEUTRAL, no change p=0.74 |
| FT / PyTorch ratio | 8.87x slower | 3.95x slower | LOSS vs PyTorch remains, gap reduced 2.25x |

PyTorch W/L/N: 0 / 1 / 0. FrankenTorch still loses this benchmark to PyTorch,
but the measured gap shrank from 8.87x to 3.95x.

## Local Stage Breakdown

| Stage | Baseline median | Candidate median | Result |
| --- | ---: | ---: | --- |
| `frankentorch_setup_tensor` | 201.30 us | 196.23 us | NEUTRAL |
| `frankentorch_forward_only` | 3.4920 ms | 3.9781 ms | LOSS, +13.9% |
| `frankentorch_sum_only` | 1.0237 ms | 1.1609 ms | LOSS, +13.4% |
| `frankentorch_backward_only` | 7.7001 ms | 4.8876 ms | WIN, -36.5%, p=0.00 |
| `kernel_forward_with_indices` | 703.76 us | 675.54 us | NEUTRAL |
| `kernel_backward_from_indices` | 1.5096 ms | 1.3126 ms | WIN, -13.1%, p=0.00 |

Stage W/L/N: 2 / 2 / 2. The end-to-end and targeted backward stage wins are
large enough to keep; the remaining losses indicate the next gap should target
forward/sum setup separately instead of widening this patch.

## Validation

All validation was run with `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b`.

| Gate | Command | Result |
| --- | --- | --- |
| Compile check | `rch exec -- cargo check -p ft-autograd --all-targets` | PASS |
| Clippy | `rch exec -- cargo clippy -p ft-autograd --all-targets -- -D warnings` | PASS |
| Autograd retain graph | `rch exec -- cargo test -p ft-autograd retain_graph_allows_second_tensor_backward --lib` | PASS |
| API max_pool3d finite diff | `rch exec -- cargo test -p ft-api functional_max_pool3d_grad_matches_finite_diff --lib` | PASS |
| Strict autograd scheduler conformance | `rch exec -- cargo test -p ft-conformance strict_scheduler_conformance_is_green --lib` | PASS |
| Whitespace | `git diff --check` | PASS |
| UBS staged-file scan | `ubs $(git diff --cached --name-only)` | FAIL: pre-existing whole-file `ft-autograd` findings outside this edit |

Workspace-wide `cargo fmt --check` was not a useful gate in this worktree:
it reports pre-existing formatting drift across unrelated examples, benches,
and non-edited regions. No formatter was applied.

UBS scans the whole staged Rust file rather than the changed hunk. It reported
existing `ft-autograd` panic/unwrap/security-heuristic inventory far away from
the `TensorBackwardReport` move hunk. `cargo check` and `cargo clippy` are green
for `ft-autograd`.
