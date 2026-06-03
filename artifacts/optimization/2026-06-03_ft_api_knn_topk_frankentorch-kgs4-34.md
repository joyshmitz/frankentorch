# ft-api KNN Exact Top-K Selection

- Bead: `frankentorch-kgs4.34`
- Parent umbrella: `frankentorch-kgs4`
- Skills: `/repeatedly-apply-skill`, `/extreme-software-optimization`, `/alien-graveyard`, `/alien-artifact-coding`
- Crate: `ft-api`
- Target benchmark: `ops_bench/knn_search/8192x512_k8`
- Outcome: keep

## Profile Target

`knn_search` returned only `k` neighbors but fully allocated and sorted all
`n_points` squared distances for every query. The target benchmark uses 8192
3D points, 512 3D queries, and `k=8`.

Baseline with rch on worker `vmi1149989`:

```text
knn_search/8192x512_k8  time: [65.592 ms 68.105 ms 70.633 ms]
```

## Lever

Replace the per-query full distance vector and stable full sort with exact
streaming top-k insertion. The retained candidate list is bounded by `k`, so the
algorithm keeps the same exhaustive distance scan but avoids `O(n log n)` full
sorting and per-query `n_points` allocation.

Alien primitive: fixed-width top-k selection / cache-local bounded candidate
set.

## Isomorphism

- Ordering: every point is still scanned in original point-index order for every
  query.
- Tie-breaking: insertion uses strict `<` only; equal distances keep the earlier
  point index ahead, matching the stable full-sort reference.
- Floating point: squared-distance arithmetic and final `sqrt` calls remain in
  the same per-candidate expression order.
- RNG: no RNG path is present.
- Golden output: `artifacts/optimization/golden_outputs/ft_api_knn_topk_frankentorch-kgs4-34.txt`
  records representative output indices and distance bit patterns.

## Result

After run with rch on worker `vmi1149989`:

```text
knn_search/8192x512_k8  time: [35.321 ms 37.579 ms 39.374 ms]
```

Delta by p50: `68.105 ms -> 37.579 ms`, about `1.81x` faster.

Score: Impact 4 x Confidence 3 / Effort 2 = 6.0.
