# frankentorch-kgs4.114 - f32 BatchNorm unit-dy reject

- Lever attempted: specialize `batch_norm_backward_f32` for exact all-ones
  upstream gradient, avoiding `dy` loads/multiplies and replacing `dbias`
  reduction with the known sample count on f32 BatchNorm training rows.
- Workload: `pytorch_gauntlet_bench`
  `gauntlet_batch_norm2d_f32_grad`, f32
  `[N,C,H,W]=[32,256,28,28]`, affine weight and bias require gradients,
  scalar `sum` loss.
- Local PyTorch oracle:
  - Active branch: FrankenTorch median `228.85 ms`, PyTorch median
    `6.8744 ms`; active FT/PyTorch ratio `33.29x` slower.
  - Disabled/final path: FrankenTorch median `238.33 ms`, PyTorch median
    `8.4699 ms`; final FT/PyTorch ratio `28.14x` slower.
  - Local current/disabled timing was noisy and not used as the keep/reject
    proof.
- Same-worker rch A/B:
  - Worker `vmi1152480`.
  - Disabled/final path median `147.30 ms`.
  - Active unit-dy branch median `157.93 ms`.
  - Active/disabled latency ratio `1.072x`; Criterion reported
    `[+1.2713% +7.2142% +13.421%]`, `p = 0.05`, performance regressed.
- Win/loss/neutral vs PyTorch: `0W / 1L / 0N`.
- Verdict: rejected and reverted from product source. The gauntlet row and
  PyTorch oracle script are kept as evidence/harness so future BatchNorm work
  can be measured directly.
- Retry condition: do not retry this exact f32 BatchNorm all-ones-`dy`
  branch. Revisit BatchNorm only with a deeper primitive that removes whole
  passes or generic train-step overhead: fused scalar-loss BatchNorm,
  saved-stat/sidecar reuse, persistent workspace or arena allocation,
  stats+backward fusion, cache-blocked per-channel reductions, or f32-native
  layout/scheduler work backed by same-worker A/B.
- Evidence:
  - `current_local_pytorch_batch_norm2d_f32.log`
  - `disabled_local_pytorch_batch_norm2d_f32.log`
  - `disabled_rch_ft_batch_norm2d_f32.log`
  - `current_rch_vmi1152480_ft_batch_norm2d_f32.log`
  - `current_rch_ft_batch_norm2d_f32.log`
  - `disabled_rch_hz2_ft_batch_norm2d_f32.log`
