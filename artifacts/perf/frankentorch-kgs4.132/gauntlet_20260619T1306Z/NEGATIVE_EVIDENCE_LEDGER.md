# frankentorch-kgs4.132 Negative-Evidence Ledger

This run keeps one measured FrankenTorch internal win and records the remaining PyTorch loss explicitly.

## Kept Lever With Residual Loss

- Lever: `apply_function_f64_borrowed_forward` in `ft-autograd`, exposed through `ft-api`, used by f64 `functional_max_pool3d` when gradients are required.
- Why this is not the prior rejected family: prior `frankentorch-kgs4.128` borrowed input slices for both forward and backward and regressed the full workload. This lever borrows only during forward setup and records an owned/context-only backward closure, so backward still uses the saved max_pool3d arg-offset sidecar instead of rereading borrowed inputs.
- Same-worker rch A/B: `8.3166 ms -> 5.4809 ms`, `1.52x` faster, Criterion p=0.00.
- Same-worker rch forward-only stage: `4.2347 ms -> 1.5978 ms`, `2.65x` faster, Criterion p=0.00.
- PyTorch ratio row: local FrankenTorch `5.4457 ms`, local PyTorch `1.6027 ms`, ratio `3.40x` slower.
- Verdict: keep source change as a measured internal win. Do not classify as PyTorch dominance.

## Loss/Neutral Rows

| Row | Evidence | Classification |
| --- | --- | --- |
| Ratio vs PyTorch | local candidate `5.4457 ms` vs PyTorch `1.6027 ms` | LOSS |
| Remote PyTorch arm | rch `hz2` failed to import torch | invalid for ratio proof |
| setup tensor stage | same-worker `236.02 us -> 234.60 us`, p=0.67 | neutral |
| sum-only stage | same-worker `1.0322 ms -> 1.0359 ms`, p=0.07 | neutral |
| backward-only stage | same-worker `15.104 ms -> 16.897 ms`, p=0.31 | neutral |
| raw forward kernel | same-worker `422.70 us -> 453.65 us`, p=0.15 | neutral |
| raw backward kernel | same-worker `999.03 us -> 1.2335 ms`, p=0.05 not below threshold | neutral/noisy |

Win/loss/neutral vs PyTorch: 0 / 1 / 0.
Stage win/loss/neutral: 1 / 0 / 5.

## Retry/Follow-Up Rule

Do not retry sidecar-only, borrowed-input-only, or unit-dout scatter variants for this workload. The next max_pool3d pass should target the remaining end-to-end gap after the forward materialization win: scalar sum/tape edge, backward scheduling, allocation churn, or a fused forward-sum-backward training primitive with fresh ratio-vs-PyTorch proof.
