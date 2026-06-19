# frankentorch-kgs4.130 max_pool3d backward overhead summary

Date: 2026-06-19
Agent: cod-b / IvoryDeer
Worktree: `/data/projects/.scratch/frankentorch-cod-b-kgs4-130-20260619T1209`

## Lever

Tried a narrow ownership-transfer lever in `TensorNodeOp::CustomFunction`
backward: when a custom-function input receives its first gradient contribution,
move the owned `Vec<f64>` returned by the backward closure into the gradient slot
instead of adding it into the zero-filled slot element-by-element.

This targeted the measured max_pool3d gap without retrying the rejected
fused-scalar, borrowed-input-only, sidecar-only, rescan-only, or standalone
unit-dout scatter branches.

## Verdict

Rejected and reverted. Product source is unchanged.

The candidate did not produce a robust keep-quality win. The directly targeted
backward-only stage was neutral, the RCH same-worker FT headline was neutral, and
the best local headline repeat remained inside the baseline interval.

## Scorecard

| Row | Baseline median | Candidate median | Result |
| --- | ---: | ---: | --- |
| Local FT headline | 15.311 ms | 15.143 ms repeat | neutral, 1.01x faster but overlapping |
| Local FT headline first candidate run | 15.311 ms | 14.757 ms | routing-only 1.04x faster, not repeated robustly |
| RCH hz2 FT headline | 27.980 ms | 28.763 ms | neutral, Criterion no-change, p=0.84 |
| Local backward-only stage | 7.7396 ms | 7.7376 ms | neutral |
| Local PyTorch ratio baseline | FT 15.311 ms / PyTorch 1.7333 ms | 8.83x slower | PyTorch win |
| Local PyTorch ratio candidate repeat | FT 15.143 ms / PyTorch 1.7762 ms | 8.53x slower | PyTorch win |

Head-to-head W/L/N vs PyTorch: 0 / 1 / 0.
Candidate A/B W/L/N: 0 / 0 / 3.

## Evidence

- `baseline_local_pytorch_criterion_max_pool3d.log`: local baseline with
  PyTorch oracle available.
- `baseline_criterion_max_pool3d.log`: RCH hz2 FT baseline; PyTorch arm failed
  because torch was unavailable on the worker.
- `baseline_stage_max_pool3d.log`: RCH hz2 stage routing evidence.
- `candidate_local_pytorch_criterion_max_pool3d.log`: local full candidate run.
- `candidate_repeat_local_ft_headline.log`: local FT headline repeat.
- `candidate_repeat_local_pytorch_headline.log`: local PyTorch headline repeat.
- `candidate_rch_criterion_max_pool3d.log`: RCH hz2 candidate run; PyTorch arm
  failed because torch was unavailable on the worker.

## Do not retry

Do not retry the first-owned-custom-gradient move as a standalone lever for this
workload. It is too shallow relative to the remaining tape/report/materialization
cost.

Next credible levers should attack a larger measured primitive: lazy/sparse
gradient slot materialization, persistent-gradient/report clone avoidance for
non-observed intermediates, or dependency scheduling that proves less allocation
and less cloning on the full max_pool3d training row.
