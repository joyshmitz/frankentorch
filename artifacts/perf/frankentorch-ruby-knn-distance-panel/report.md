# frankentorch-swbh KNN distance-panel rejection

## Target

- Bead: `frankentorch-swbh`
- Benchmark: `knn_search/8192x512_k8`
- Crate: `ft-api`
- Worker: `vmi1293453`

This pass followed the profile-backed residual from `frankentorch-tgst`: after
fixed-buffer top-k, the remaining KNN work still scans every point/query pair in
an exact 16-query tile.

## Candidate

Tested one exact dimension-level threshold lever:

- once a query tile's top-k buffer is full, compute `dx^2`;
- reject the point for that query if `dx^2` is not strictly less than the
  current worst retained squared distance;
- otherwise compute `dy^2`, reject on `dx^2 + dy^2` when that partial sum cannot
  enter the top-k;
- otherwise compute the original full squared distance and call the existing
  strict-less top-k insertion helper.

No approximate nearest-neighbor index, point reordering, query reordering, or
distance formula replacement was used.

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

`rch` selected `vmi1293453`.

```text
knn_search/8192x512_k8  time: [7.1776 ms 7.4241 ms 7.7556 ms]
```

## Proof

Focused bit-exact KNN proof on `vmi1293453`:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-api knn_search -- --nocapture
```

Result:

```text
running 3 tests
test tests::knn_search_zero_k_returns_empty_outputs ... ok
test tests::knn_search_streaming_topk_matches_full_sort_reference_bit_exact ... ok
test tests::knn_search_bench_scale_matches_full_sort_reference_bit_exact ... ok
test result: ok. 3 passed; 0 failed; 1922 filtered out
```

Isomorphism ledger:

- Ordering preserved: yes. Query tiles, local query order, and ascending point
  scan order were unchanged.
- Tie-breaking unchanged: yes. The gate only rejected candidates that were not
  strictly less than the current worst retained squared distance, matching the
  existing full-buffer helper.
- Floating point: retained candidate distances used the same left-associative
  `powi(2)` squared-distance expression and same final `sqrt`; skipped
  candidates were provably unable to enter the output set.
- RNG: none.
- Golden output: bench-scale full-sort reference digest test passed unchanged.

## Rebenchmark

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1293453 rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

```text
knn_search/8192x512_k8  time: [7.4549 ms 7.7303 ms 7.9304 ms]
```

Median ratio: `7.4241 / 7.7303 = 0.960x`.

## Verdict

Rejected. The exact partial-distance threshold lever regressed on the same
worker, so the source hunk was removed and no KNN code was kept.

Score: `Impact 0.96 x Confidence 0.95 / Effort 1.0 = 0.91`, below the `2.0`
keep gate.

Next primitive: do not repeat scalar partial-threshold pruning. Re-profile and
route either to a materially different exact KNN distance-panel/SIMD layout or a
different profile-backed hotspot.
