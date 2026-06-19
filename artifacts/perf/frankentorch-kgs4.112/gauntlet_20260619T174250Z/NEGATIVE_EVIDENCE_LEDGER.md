# frankentorch-kgs4.112 negative-evidence ledger

- Bead: `frankentorch-kgs4.112`
- Agent: `IvoryDeer`
- Date: 2026-06-19
- Scope: f64 avg_pool2d 2x2 stride-2 no-padding backward, training-style scalar
  sum backward.

## Result

- Existing code-first fast path baseline: rch `hz2`
  `avg_pool2d/grad` median `58.600 ms`.
- Direct-assignment candidate: rch `hz2` median `68.624 ms`, Criterion
  `[+4.6329% +13.137% +24.143%]`, `p=0.01`.
- Candidate verdict: rejected and reverted.
- PyTorch head-to-head: local FrankenTorch median `16.627 ms`; local PyTorch
  `2.12.0+cpu` median `3.6632 ms`; FrankenTorch is `4.54x` slower.
- Win/loss/neutral vs PyTorch: `0W / 1L / 0N`.

## What Not To Retry

Do not retry replacing the current guarded non-overlap scatter `+=` writes with
direct assignment writes for this row. Do not spend another pass on a tiny local
scatter branch unless a fresh profile proves `avg_pool2d_backward_2x2s2_f64`
dominates after tape setup, allocation churn, scalar sum backward, and tensor
materialization are separated.

## Remaining Route

The remaining gap is not closed by local scatter mechanics. Route next work to
end-to-end tape/allocation/sum-backward overhead, native f32 layout, or a fused
avg-pool training primitive with fresh ratio-vs-PyTorch proof.

## Evidence

- `baseline_rch_ops_avg_pool2d_grad.log`: rch `hz2`, current median `58.600 ms`.
- `candidate_rch_ops_avg_pool2d_grad.log`: rch `hz2`, candidate median
  `68.624 ms`, statistically significant regression.
- `generic_disabled_rch_ops_avg_pool2d_grad.log`: rch `ovh-b`, generic-disabled
  routing row median `117.51 ms`; cross-worker only.
- `local_pytorch_gauntlet_avg_pool2d.log`: local FT/PyTorch ratio row.
- `rch_pytorch_gauntlet_avg_pool2d.log`: remote FT build/bench row; PyTorch arm
  failed with missing `torch`.
- `test_ft_conformance.log`: conformance green.
