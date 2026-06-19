# frankentorch-nzqb9 Scorecard

Date: 2026-06-19
Agent: IvoryDeer (cod-b)

## Decision

No source change kept.

Three bold levers were tried and reverted because none produced a credible full-workload win:

| Lever | Target row | Result | Decision |
| --- | ---: | --- | --- |
| Scalar `Sum` backward direct accumulation | `sum_only` | 997.97 us -> 998.70 us, p=0.93 | Revert |
| Power-of-two exact pairwise sum | `sum_only` | 997.97 us -> 1.0481 ms, p=0.89 | Revert |
| CustomFunction single-contribution move | full row | 6.4150 ms -> 6.1558 ms, p=0.22 | Revert |

## PyTorch Ratio

Local PyTorch-enabled row from current head:

| Runtime | Median |
| --- | ---: |
| FrankenTorch | 7.3569 ms |
| PyTorch 2.12 CPU | 1.7639 ms |

Ratio: FrankenTorch is 4.17x slower.

Win/loss/neutral vs PyTorch: 0W / 1L / 0N.

## Notes

- RCH workers used for FT timing lack `torch`; remote PyTorch rows fail with `ModuleNotFoundError`.
- Remote same-worker stage proof used `hz2`.
- Local PyTorch ratio used `/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python`.
- Conformance: no product source was kept, so no conformance gate is required for a shipped code change. Focused candidate tests were run while candidates were live and are retained in the evidence logs.

## Next Route

The remaining gap is not closed by local sum/scatter micro-branches. The next credible attempt should change the full tape cost model: lazy gradient slot allocation, arena reuse for backward buffers, or a fused `max_pool3d -> sum -> backward` primitive that removes both the dense sum gradient and the second 1M-element input-gradient buffer in one measured full-row lever.
