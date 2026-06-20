# frankentorch-kgs4.119 negative-evidence ledger

Date: 2026-06-20
Agent: IvoryDeer
Worktree: `/data/projects/.scratch/frankentorch-cod-b-kgs4-119-20260620T1112`
Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b`

## Lever

`ft-api` already routes the f64 `functional_conv3d` custom autograd path through
`apply_function_with_create_graph_borrowed_inputs`, so the backward closure borrows
the padded input and weight instead of replaying `ctx.save_for_backward(pv.to_vec())`
and `ctx.save_for_backward(wv.to_vec())`. This verifies the code-first lever from
`231dbe98` and rejects reverting it to the saved-copy path.

Bench shape for the PyTorch gauntlet row:

- input: `[2, 32, 8, 16, 16]`, f64, requires grad
- weight: `[32, 32, 3, 3, 3]`, f64, requires grad
- stride: `(1, 1, 1)`
- padding: `(1, 1, 1)`
- loss: `conv3d(...).sum().backward()`

## Result ledger

| Result | Evidence | Host | Median | Ratio vs PyTorch | Verdict |
| --- | --- | --- | ---: | ---: | --- |
| FrankenTorch current borrowed-input | `current_local_gauntlet_conv3d_pytorch.log` | local | `24.095 ms` | `2.38x` slower than PyTorch | PyTorch loss |
| PyTorch `2.12.1+cpu` | `current_local_gauntlet_conv3d_pytorch.log` | local | `10.126 ms` | `1.00x` | Baseline |
| FrankenTorch current borrowed-input | `current_rch_retry_gauntlet_conv3d_ft.log` | `ovh-a` | `15.632 ms` | N/A, no Torch on rch worker | Internal win comparator |
| FrankenTorch disabled save-copy variant | `disabled_rch_vmi1152480_gauntlet_conv3d_ft.log` | `ovh-a` | `19.429 ms` | N/A, no Torch on rch worker | Internal baseline |
| FrankenTorch current borrowed-input | `current_rch_vmi1152480_gauntlet_conv3d_ft.log` | `vmi1152480` | `28.364 ms` | N/A, no Torch on rch worker | Routing-only neutral |

Same-worker A/B: `19.429 / 15.632 = 1.24x` faster for the borrowed-input path on
`ovh-a`. Criterion also reported `[-27.005%, -22.256%, -14.205%]`, `p = 0.00`, for
the current rerun after the disabled variant. The initial `vmi1152480` current-only
row is not used for keep/reject because the disabled comparison landed on `ovh-a`.

W/L/N vs PyTorch for this bead: `0 / 1 / 0`.

## Decision

KEEP the borrowed-input implementation. It is still a PyTorch loss on the local
head-to-head row, but the same-worker A/B shows reverting to saved copies would be
a real regression.

Next route: the remaining `2.38x` PyTorch gap is not the saved-input clone. Move to
whole-row autograd/tape allocation, sum-loss gradient materialization, persistent
workspaces, direct fused conv3d sum-backward, or a fundamentally different layout
strategy with same-worker proof.
