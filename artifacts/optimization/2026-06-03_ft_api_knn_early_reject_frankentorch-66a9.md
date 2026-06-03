# ft-api KNN Top-K Early Reject

- Bead: `frankentorch-66a9`
- Target benchmark: `ops_bench/knn_search/8192x512_k8`
- Pass: 3 of repeated optimization loop
- Lever: explicit `Vec<(usize, f64)>` bounded top-k buffer plus early reject when
  `best.len() == k && d.partial_cmp(&best[k - 1].1) != Some(Less)`.

## Profile Target

`knn_search` still computes exact squared distances for all 8192 points across
512 queries and returns `k=8`. After the previous top-k pass, the remaining hot
inner-loop waste was scanning the retained `k` candidates for point distances
that cannot enter the bounded result set.

Fresh same-worker baseline on `vmi1156319`:

```text
knn_search/8192x512_k8  time: [52.031 ms 54.826 ms 60.539 ms]
```

## Alien Primitive Mapping

- Primary primitive: cache-local bounded candidate set / exact fixed-width
  top-k filter. The candidate set stays at `k=8`; the current worst retained
  distance is a deterministic admission threshold.
- Graveyard anchor: `alien_cs_graveyard.md` section 4.6
  `Nested Data Parallelism / Flattening Transform` covers segmented arrays and
  segment-aware filtering. This KNN loop is the scalar per-query version:
  each query owns one segment-local top-k buffer.
- Graveyard anchor: `alien_cs_graveyard.md` section 5.6
  `Polyhedral Compilation / Loop Transformations` motivates locality-preserving
  loop scheduling. This pass does not tile or reorder; it only removes redundant
  work inside the existing affine query-point scan.
- Related deeper anchor: `alien_cs_graveyard.md` section 7.11
  `Approximate Nearest Neighbor Indexes` describes candidate generation before
  exact rerank. That is not used here because this bead requires exact output.

## Opportunity Score

Impact 3 x Confidence 4 / Effort 1 = 12.0.

The score is high because the lever is local, does not alter traversal order,
and targets a measured inner-loop cost after the first exact top-k rewrite.

## Isomorphism Proof

- Ordering preserved: yes. Query order, point scan order, and retained output
  order are unchanged.
- Tie-breaking unchanged: yes. Only distances strictly less than the current
  worst retained distance may enter a full buffer; equal or unordered distances
  are rejected just as the insertion scan would fail to place them before the
  existing candidate.
- Floating-point unchanged: yes. The same candidate distance is computed before
  the threshold check; no distance arithmetic is reordered or approximated.
- RNG unchanged: yes. `knn_search` has no RNG path.
- Golden output: pass-specific checksum
  `4688c9e7ee906524b0d77e7fba72d94c9c939d554ad5aa15d1b439af33aed63f`.

## Validation

- Focused remote proof on `vmi1156319`:
  `cargo test -p ft-api knn_search_streaming_topk_matches_full_sort_reference_bit_exact -- --nocapture`
- Golden checksum: passed for the pass-specific output above.

## Result

Same-worker after benchmark on `vmi1156319`:

```text
knn_search/8192x512_k8  time: [17.349 ms 17.908 ms 18.248 ms]
```

Delta by p50: `54.826 ms -> 17.908 ms`, about `3.06x` faster.

## Next Primitive After Re-profile

Re-profile before changing code again. The next deeper exact primitive is
cache-tiled segmented top-k: process query tiles against point tiles while
keeping one stable bounded buffer per query segment. That would preserve
point-order semantics while improving point-coordinate reuse across queries.
