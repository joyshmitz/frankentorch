# ft-api SupCon Symmetric Similarity Attempt - frankentorch-y0gy

## Target

- Bead: `frankentorch-y0gy`
- Crate: `ft-api`
- Benchmark: `supcon_loss/512x512`
- Baseline command: `rch exec -- cargo bench -p ft-api --bench ops_bench -- supcon_loss/512x512 --warm-up-time 1 --measurement-time 5 --sample-size 20`
- Baseline worker: `ts1`
- Baseline: `[9.0834 ms 9.4931 ms 10.127 ms]`

## Attempted Lever

The candidate computed only the upper triangle of the SupCon normalized similarity matrix, then mirrored the lower triangle before the existing loss reduction. The intended win was to cut duplicate `(i,j)` / `(j,i)` dot products while preserving per-dot `k` summation order and row-wise loss traversal.

## Proof Result

- Focused proof: `rch exec -- cargo test -p ft-api supcon_loss_parallel_match_serial_bit_exact -- --nocapture` passed on `ts2`.
- Isomorphism checked: scalar loss remained bit-identical to the full serial matrix reference for the parallel-path fixture.
- Source status: candidate code manually reverted after the benchmark failed the keep threshold.

## Benchmark Result

- Candidate command: `rch exec -- cargo bench -p ft-api --bench ops_bench -- supcon_loss/512x512 --warm-up-time 1 --measurement-time 5 --sample-size 20`
- Candidate worker: `ts2`
- Candidate: `[9.2158 ms 9.3131 ms 9.4556 ms]`
- Cross-worker p50 ratio: `9.4931 ms / 9.3131 ms = 1.020x`
- Score: `Impact 1.02 * Confidence 0.75 / Effort 1.0 = 0.765`
- Verdict: reject.

## Pivot

This was the wrong depth of lever for SupCon: halving dot products did not dominate enough relative to row scheduling, normalization, loss reduction, and memory traffic. Next pass should attack a different profile-backed primitive, preferably a separable interpolation or normalization kernel with reusable per-axis weights/indices while preserving arithmetic order.
