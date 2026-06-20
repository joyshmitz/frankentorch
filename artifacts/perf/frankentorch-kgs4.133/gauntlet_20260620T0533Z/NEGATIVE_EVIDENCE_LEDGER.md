# frankentorch-kgs4.133 Negative Evidence

- Date: 2026-06-20
- Candidate: f64 `conv2d_backward_f64` all-ones-`dout` row-collapse branch
- Baseline comparator: existing generic f64 conv2d backward path
- PyTorch comparator: local CPU PyTorch `2.12.1+cpu`

## Measurements

- Current FrankenTorch, same-worker rch `vmi1152480`: `121.07 ms`
- Active candidate, same-worker rch `vmi1152480`: `117.92 ms`
- Criterion change interval: `[-7.9705% -2.5970% +2.8489%]`
- Significance: `p = 0.38`; no change detected
- PyTorch median: `63.449849 ms`
- Current FT/PyTorch ratio: `1.91x` slower
- Candidate FT/PyTorch ratio: `1.86x` slower

## Decision

Rejected and removed from product source. The candidate did not clear the
same-worker keep gate, and the row remains a PyTorch loss.

## Retry Condition

Retry conv2d only when fresh profiling supports a different primitive from this
one. Do not retry all-ones row collapse while still materializing the full im2col
panel and ones-vector GEMMs.
