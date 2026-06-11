# Pass 4 - Eigvals Column-Start Range Cut Rejected

Date: 2026-06-11
Bead: `frankentorch-npxbw`

## Lever

Tried an eigvals-only Francis QR range cut: when `want_vectors == false`, start
the bulge column update at the active block top `l` rather than row `0`, because
rows above `l` are upper-block coupling after the split and should not feed active
block eigenvalues.

During RCH validation a row-split rewrite appeared in the same row/column hunk,
so the current candidate was evaluated as the combined local hunk and then
removed after the performance gate failed. No source change is kept.

## Behavior Proof

- Focused eig/eigh tests passed: `cargo test -p ft-kernel-cpu --lib eig -- --nocapture`
  on RCH, `21 passed`.
- Strict `eigvals_golden` SHA stayed
  `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.
- Strict diff against pass 1 was empty.

## Benchmark

Same-worker `vmi1227854` Criterion row:

| Row | Baseline/current before lever | Candidate |
| --- | --- | --- |
| `eigvals_f64_256x256` | `[25.052 ms 25.370 ms 25.694 ms]` | `[25.565 ms 25.806 ms 26.046 ms]` |

## Verdict

Rejected. Score is `0`: behavior was preserved, but the same-worker benchmark
regressed. Next pass should avoid range micro-cuts and move to the deeper direct
small-bulge/multishift primitive or a measurable sweep-count/far-update lever.
