# ft-api KNN Query Tiling

- Bead: `frankentorch-66a9`
- Target benchmark: `ops_bench/knn_search/8192x512_k8`
- Pass: 5 of repeated optimization loop
- Lever: exact 16-query cache tile over the point sweep, retaining one stable
  bounded top-k buffer per query.

## Profile Target

Fresh rch Criterion baseline on `vmi1149989` before this lever:

```text
knn_search/8192x512_k8  time: [20.926 ms 22.766 ms 24.449 ms]
```

The target still performs `512 * 8192` exact squared-distance evaluations.
The previous pass reduced top-k insertion cost; the remaining profiler-evident
waste is reloading the same point coordinates independently for every query.

## Alien Primitive Mapping

- Graveyard primitive: `alien_cs_graveyard.md` section 5.6, polyhedral loop
  transformations. This pass performs a legality-preserving query/point loop
  interchange inside fixed query tiles.
- Graveyard primitive: `alien_cs_graveyard.md` section 8.2, vectorized
  execution and morsel-driven processing. The 16-query tile is a cache-sized
  morsel: one point load feeds multiple independent query segments.
- Rejected alternative: section 7.11 approximate nearest-neighbor indexes.
  This bead requires exact PyTorch-observable ordering, so ANN candidate
  pruning is out of scope for this lever.

## Opportunity Score

Impact 5 x Confidence 4 / Effort 1 = 20.0.

The score clears the keep threshold because the lever is local, exact, and
measured as a same-worker 1.80x p50 improvement.

## Isomorphism Proof

- Ordering preserved: yes. Output queries are emitted in ascending query order.
  For each query, candidate points are still considered in ascending point
  order.
- Tie-breaking unchanged: yes. Each query uses the same strict-less insertion
  comparator and never admits equal or unordered distances into a full buffer.
- Floating-point unchanged: yes. Each candidate distance uses the same
  `(px - qx).powi(2) + (py - qy).powi(2) + (pz - qz).powi(2)` expression and
  the same final `sqrt`; only independent query work is interleaved.
- RNG unchanged: yes. `knn_search` has no RNG path.
- Golden output: pass-specific checksum
  `1d68ca23ce9cc599bbd007ced82e4649490891d5edefbedfcda7647b6399f33a`
  for `ft_api_knn_search_frankentorch-66a9-pass5.txt`; the bench-scale proof
  test also checks full-sort equality and output bit digests.

## Validation

- Focused remote proof on `vmi1149989`:
  `cargo test -p ft-api knn_search -- --nocapture` passed 3 KNN tests.
- Re-benchmark on `vmi1149989`:

```text
knn_search/8192x512_k8  time: [12.181 ms 12.641 ms 13.177 ms]
```

Delta by p50: `22.766 ms -> 12.641 ms`, about `1.80x` faster.

## Next Primitive After Re-profile

Re-profile before changing code again. The next exact primitive should attack
the current post-tiling hotspot rather than adding another scalar tweak.
