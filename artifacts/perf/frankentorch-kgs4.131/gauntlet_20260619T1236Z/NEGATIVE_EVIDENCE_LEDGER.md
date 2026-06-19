# frankentorch-kgs4.131 Negative Evidence Ledger

## Residual PyTorch Loss

FrankenTorch remains slower than PyTorch on `gauntlet_max_pool3d_grad`.

| Snapshot | FrankenTorch median | PyTorch median | Ratio |
| --- | ---: | ---: | ---: |
| Baseline local paired run | 15.718 ms | 1.7717 ms | 8.87x slower |
| Candidate local paired run | 6.6743 ms | 1.6898 ms | 3.95x slower |

Decision: keep the lever because the gap shrank materially, but do not mark the
PyTorch gap closed. The next bead should attack the remaining forward/sum and
autograd setup costs.

## Remote PyTorch Arm Failure

Both rch runs selected `hz1` and produced directly comparable FrankenTorch A/B
timings, but the PyTorch arm failed remotely:

```text
ModuleNotFoundError: No module named 'torch'
```

Decision: use the rch run only for same-worker FrankenTorch A/B evidence, and
use the local PyTorch-enabled run for FT-vs-PyTorch ratio evidence.

## Stage Regressions

The local stage breakdown still contains losses:

| Stage | Baseline median | Candidate median | Result |
| --- | ---: | ---: | --- |
| `frankentorch_forward_only` | 3.4920 ms | 3.9781 ms | LOSS, +13.9% |
| `frankentorch_sum_only` | 1.0237 ms | 1.1609 ms | LOSS, +13.4% |

Decision: do not revert because headline and backward-stage wins dominate for
the target benchmark. Route follow-up work to forward/sum setup rather than
expanding this autograd report-materialization patch.

## Non-actionable Formatter Noise

`cargo fmt --check` was stopped after it emitted large diffs for unrelated
pre-existing formatting drift across examples, benches, and non-edited code
regions.

Decision: do not format unrelated files. `git diff --check` passed for the
actual patch.
