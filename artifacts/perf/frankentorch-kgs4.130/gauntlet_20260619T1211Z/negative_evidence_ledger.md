# Negative-evidence ledger: frankentorch-kgs4.130

## 2026-06-19 - MaxPool3d custom-gradient ownership transfer

- Lever: move the first owned custom-function input gradient into the tape
  gradient slot instead of accumulating it into a zero-initialized slot.
- Workload: `gauntlet_max_pool3d_grad`, deterministic f64
  `[N,C,D,H,W]=[2,32,16,32,32]`, kernel `2x2x2`, stride `2x2x2`,
  forward max_pool3d, scalar `sum`, backward.
- Reference: local PyTorch `2.12.1+cpu` via
  `/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python`.
- Local baseline: FrankenTorch median `15.311 ms`; PyTorch median `1.7333 ms`;
  ratio vs PyTorch `8.83x` slower.
- Candidate repeat: FrankenTorch median `15.143 ms`; PyTorch median
  `1.7762 ms`; ratio vs PyTorch `8.53x` slower.
- RCH same-worker FT-only row on `hz2`: baseline `27.980 ms`; candidate
  `28.763 ms`; Criterion reported no change, p=0.84. The PyTorch arm failed on
  the worker with `ModuleNotFoundError: No module named 'torch'`, so the worker
  row is FT-only A/B evidence, not ratio proof.
- Targeted local backward-only stage: baseline `7.7396 ms`; candidate
  `7.7376 ms`; neutral.
- Head-to-head W/L/N vs PyTorch: `0/1/0`.
- Candidate A/B W/L/N: `0/0/3`.
- Verdict: rejected and reverted. No product source kept.
- Retry condition: do not retry the first-owned-custom-gradient move by itself.
  Revisit max_pool3d backward overhead only with a larger measured reduction in
  tape/report/materialization cost, such as lazy gradient slots or avoiding
  persistent/report clones for unobserved intermediates while preserving public
  graph semantics.
