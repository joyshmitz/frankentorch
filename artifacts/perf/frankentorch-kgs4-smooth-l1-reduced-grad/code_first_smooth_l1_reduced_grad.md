# frankentorch-kgs4.124: SmoothL1 Direct Reduced Grad

## Lever

Direct f64 same-shape `tensor_smooth_l1_loss(..., "mean" | "sum", beta)` through a scalar custom autograd op when either input requires grad.

The previous grad path built a full per-element SmoothL1 tensor, then added a `tensor_mean` or `tensor_sum` node. Backward therefore materialized a uniform `dloss` vector before calling `smooth_l1_backward_f64`. This pass computes the scalar reduced value directly via the existing fused reducer and backpropagates the scalar upstream scale through `smooth_l1_backward_reduced_f64`.

## Guards

- Kernel guard: `smooth_l1_reduced_backward_f64_matches_uniform_dloss_bits` compares reduced scalar backward to the old materialized uniform-`dloss` helper bit-for-bit.
- API guard: `smooth_l1_loss_reduced_grad_matches_none_then_reduce_bits` compares direct reduced autograd against `reduction="none"` followed by the existing reduction graph for output, input grad, and target grad bits.

## Negative-Evidence Ledger

| Attempt | Evidence | Outcome | Retry rule |
| --- | --- | --- | --- |
| f32 SmoothL1 no-grad fused path | `artifacts/perf/frankentorch-cs2d/rejected_f32_smooth_l1_fast_path.md` | Rejected/no stable after benchmark; mixed f32/f64 constants and artifact churn blocked a keep. | Do not retry f32 no-grad SmoothL1 without a fresh dtype audit and same-worker A/B. |
| f64 SmoothL1 no-grad pairwise reducer | `frankentorch-lonz`, `artifacts/perf/frankentorch-ruby-smoothl1-f64-reduction/report.md` | Kept; baseline 136.80 ms -> 97.302 ms. | Do not rework the no-grad reducer family; this bead is grad-only. |
| direct reduced Gaussian NLL grad | `frankentorch-fdn1v` | Rejected; same-worker median regressed 829.27 ms -> 1.0274 s. | Do not generalize this SmoothL1 lever to Gaussian NLL without new profile proof. |
| SmoothL1 direct reduced grad | `gauntlet_20260619/` logs below | Kept as a FrankenTorch internal win: same-worker `hz2` Criterion improved `smooth_l1/grad_8m` from 963.16 ms to 757.63 ms median (`1.27x`). Still a PyTorch loss: local current FT 742.95 ms vs PyTorch 373.61 ms (`1.99x` slower). | Do not retry another scalar-reduction wrapper. Next route is deeper tape/RNG/loss-kernel/SIMD work until this row beats PyTorch. |

## Proof Status

Gauntlet measured on 2026-06-19; no source revert.

### Criterion / PyTorch Evidence

| Row | Host/worker | Median | Evidence |
| --- | --- | ---: | --- |
| Pre-lever FT `81032a4d` `smooth_l1/grad_8m` | `hz2` via `RCH_WORKER=hz2` baseline worktree | 963.16 ms | `gauntlet_20260619/prelever_81032a4d_criterion_smooth_l1_grad_8m.log` |
| Current FT `smooth_l1/grad_8m` | `hz2` via `RCH_WORKER=hz2` | 757.63 ms | `gauntlet_20260619/current_hz2_criterion_smooth_l1_grad_8m.log` |
| Current FT `smooth_l1/grad_8m` | local host Criterion | 742.95 ms | `gauntlet_20260619/current_local_criterion_smooth_l1_grad_8m.log` |
| PyTorch original `smooth_l1_loss(...).backward()` | local host, PyTorch 2.12.1 CPU path, 32 threads | 373.61 ms | `gauntlet_20260619/torch_smooth_l1_grad_8m_local.json` |

Verdict: keep the lever because the decisive same-worker FT A/B is a non-overlapping `1.27x` speedup. Do not count this as upstream dominance: local FT/PyTorch time ratio is `742.95 / 373.61 = 1.99x`, so PyTorch is still about `2.0x` faster on the realistic 8M f64 mean-loss backward row.

Supplemental drift row: unpinned current FT on `ovh-a` measured 595.82 ms median in `gauntlet_20260619/current_criterion_smooth_l1_grad_8m.log`; it is not used for the keep/reject comparison because the pre-lever row ran on `hz2`.

Tracker: `frankentorch-kgs4.124` closed as measured keep. Successor `frankentorch-kgs4.128` tracks the remaining SmoothL1-vs-PyTorch gap.

### Guards

- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo test -p ft-kernel-cpu smooth_l1_backward_reduced_f64_matches_uniform_dloss_bits -- --nocapture`: passed; log `gauntlet_20260619/ft_kernel_cpu_smooth_l1_guard.log`.
- Initial `ft-api` guard exposed unrelated stale default-argument test calls for `tensor_celu`, `tensor_softshrink`, and `tensor_diagflat`; those calls were updated to explicit PyTorch defaults.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo test -p ft-api smooth_l1_loss_reduced_grad_matches_materialized_reference_bits -- --nocapture`: retry passed; log `gauntlet_20260619/ft_api_smooth_l1_guard_retry1.log`.
