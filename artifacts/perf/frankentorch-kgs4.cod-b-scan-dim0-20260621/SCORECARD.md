# frankentorch-kgs4.cod-b scan dim0 scorecard

Date: 2026-06-21
Agent: IvoryDeer
Target: cumsum/cumprod f64 dim0 forward+backward `[262144,64]`

## Lever

Tried a no-zero fast path for `cumprod_backward_tensor_contiguous_f64`: scan each
row-contiguous lane in reverse, keep the same per-lane accumulation order, and
fall back to the generic implementation if any input has `abs() <= f64::EPSILON`.

## Result

Verdict: rejected and reverted.

The baseline run used the stale shared checkout on RCH `hz2`; remote PyTorch was
unavailable because the worker lacks `torch`.

| Workload | FT baseline | Local PyTorch | Mixed-location ratio |
|---|---:|---:|---:|
| `cumsum` fwd+bwd dim0 | 659.333 ms | 495.924797 ms | FT 1.33x slower |
| `cumprod` fwd+bwd dim0 | 965.653 ms | 545.352066 ms | FT 1.77x slower |

The candidate run moved to RCH `vmi1153651`. The unchanged `cumsum` control row
measured 1323.053 ms and the targeted `cumprod` row measured 2135.452 ms; remote
PyTorch was still unavailable. RCH reported `RCH-E309` after the successful
remote run because artifact retrieval failed, so the local build artifacts were
incomplete.

Because the control row moved roughly 2x slower on a different worker and there
was no same-worker proof, the result is not keep evidence. The temporary source
hunk was reverted and no product code was committed.

## Gates

`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b AGENT_NAME=IvoryDeer rch exec -- cargo test -p ft-conformance --profile release`
passed on RCH `ovh-a`: `ft-conformance` release-profile tests passed, including
the 199-test library suite plus binaries, integration tests, smoke tests, and
doc tests.

## Ratio ledger

This pass: `0W / 2L / 0N` as mixed-location routing evidence only.

Retry condition: revisit only from current `origin/main` with same-worker FT
baseline/candidate proof and a PyTorch-capable comparator, or if a future
profile isolates `cumprod` backward as the remaining current-origin loss after
the existing scan and batched-linalg wins.
