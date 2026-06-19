# frankentorch-nzqb9 Negative Evidence Ledger

Date: 2026-06-19
Agent: IvoryDeer (cod-b)
Worktree: /data/projects/.scratch/frankentorch-cod-b-bold-verify-20260619T1349Z
Target: ft-api gauntlet max_pool3d grad, f64 [2,32,16,32,32], kernel 2x2x2, stride 2x2x2, forward max_pool3d -> scalar sum -> backward.

## Baseline

- Remote FT full row on rch `ovh-a`: 4.0368 ms. PyTorch row failed on worker because `torch` is not installed.
- Remote FT stage baseline on rch `hz2`:
  - setup: 227.46 us
  - forward_only: 1.8538 ms
  - sum_only: 997.97 us
  - backward_only: 17.612 ms
  - kernel_forward_with_indices: 432.50 us
  - kernel_backward_from_indices: 1.0809 ms
- Local PyTorch-enabled current head ratio row:
  - FrankenTorch: 7.3569 ms
  - PyTorch 2.12 CPU: 1.7639 ms
  - ratio: 4.17x slower

## Rejected Levers

1. Scalar `Sum` backward direct accumulation.
   - Change tried: replace `vec![grad_scalar; input_numel]` plus accumulation with lazy scalar accumulation into the input grad slot.
   - Same-worker rch `hz2` stage result:
     - sum_only: 997.97 us -> 998.70 us, neutral, p=0.93.
     - backward_only: 17.612 ms -> 14.651 ms, neutral/noisy, p=0.52.
   - Full row on rch `hz2`: 6.4150 ms; PyTorch row unavailable on worker.
   - Verdict: rejected and reverted. The scalar-sum tape edge is not moved enough by this local allocation removal.

2. Power-of-two exact pairwise sum fast path.
   - Change tried: replace recursive pairwise sum for power-of-two reductions with an iterative 128-leaf tree that preserves the same tree order.
   - Correctness probe passed while candidate was live: `sum_power2_block128_fast_path_matches_recursive_tree`.
   - Same-worker rch `hz2` stage result:
     - sum_only: 997.97 us -> 1.0481 ms, neutral/regressive, p=0.89.
     - backward_only: 17.612 ms -> 14.729 ms, neutral/noisy, p=0.82.
   - Verdict: rejected and reverted. Recursion overhead is not the measured sum bottleneck.

3. CustomFunction single-contribution move.
   - Change tried: when a custom backward produced the only dependency edge for an input, move its gradient Vec into the target slot instead of adding into a pre-zeroed slot; guarded `-0.0` to preserve `0.0 + grad` bits.
   - Correctness probe passed while candidate was live: `custom_function_single_contribution_preserves_negative_zero_accumulate_bits`.
   - Same-worker rch `hz2` stage result:
     - backward_only: 17.612 ms -> 12.411 ms by p50, but Criterion reported no statistically significant change, p=0.19, confidence interval crossed regression.
   - Full row on rch `hz2`: 6.4150 ms -> 6.1558 ms, neutral, p=0.22.
   - Verdict: rejected and reverted. The full training row does not keep the apparent backward-only p50 gain.

## Summary

- Kept source changes: none.
- Reverted source trials: 3/3.
- Win/loss/neutral vs PyTorch: 0W / 1L / 0N.
- Current local PyTorch ratio: FrankenTorch 4.17x slower.
- Retry condition: do not retry local-only scalar-sum accumulation, recursive-pairwise replacement, sidecar-only, borrowed-input-only, unit-dout scatter, or single-contribution move variants for this workload. The next credible route needs a broader lazy gradient storage/arena change that avoids initial zero allocation and second full-size buffers across the whole tape, or a fused `max_pool3d -> sum -> backward` training primitive with fresh same-worker full-row proof.

## Evidence Files

- `baseline_rch_max_pool3d.log`
- `baseline_rch_max_pool3d_stage.log`
- `after_rch_max_pool3d_stage.log`
- `after_rch_max_pool3d.log`
- `after_rch_max_pool3d_stage_sum_power2.log`
- `test_sum_power2_fast_path.log`
- `after_rch_max_pool3d_stage_custom_move.log`
- `after_rch_max_pool3d_custom_move.log`
- `test_custom_single_contribution_negative_zero.log`
- `local_pytorch_ratio_max_pool3d.log`
