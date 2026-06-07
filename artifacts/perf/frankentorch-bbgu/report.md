# frankentorch-bbgu exact KNN distance microkernel

## Pass 1 baseline/profile

Fallback selection: `br ready --json` returned `[]` twice after `frankentorch-8x2i`, and the only open perf lanes were claimed by other agents (`frankentorch-rd1s` by BlackThrush, `frankentorch-kgs4` by RusticKite). This pass used the existing profile-backed KNN lane:

- `frankentorch-tgst` kept fixed-size top-k maintenance for `knn_search/8192x512_k8` (`9.5135 ms -> 7.6896 ms`, Score `2.61`).
- `frankentorch-swbh` rejected scalar partial-distance threshold pruning on the same exact KNN benchmark (`7.4241 ms -> 7.7303 ms`, Score `0.91`).
- The next primitive called for a materially different exact distance-panel/SIMD-style layout or distance microkernel, not another scalar pruning lever.

Initial baseline command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts1`

Criterion:

```text
knn_search/8192x512_k8  time: [8.0860 ms 8.2396 ms 8.3942 ms]
```

During the focused proof run the shared branch advanced to `a4aee7ad`, which already contained the exact `dx * dx + dy * dy + dz * dz` distance expression. That made the initial baseline stale for scoring this pass's remaining source delta.

Matched clean baseline for the actual candidate was taken from detached worktree `/data/projects/.scratch/frankentorch-bbgu-baseline-a4aee7ad` at `a4aee7ad`:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts1`

```text
knn_search/8192x512_k8  time: [8.2331 ms 8.5562 ms 9.0302 ms]
```

## Pass 2-3 primitive and proof plan

One lever only: increase the exact KNN query panel from `16` to `32` queries after the shared branch already incorporated the `dx * dx` distance microkernel.

This is not the rejected `swbh` partial-threshold/pruning family. It keeps the same exhaustive point/query scan and strict top-k insertion contract, but tests whether a wider query tile improves point reuse and loop amortization.

Isomorphism obligations:

- API/ABI preserved: same `knn_search(points, queries, k)` signature, output shapes, and errors.
- Ordering preserved: batch order, query-tile order, point scan order, local query order, and output write order unchanged.
- Tie-breaking unchanged: `consider_knn_candidate` still uses strict `<`; equal distances keep earlier point order.
- Floating-point drift gate: candidate changes only panel width. The focused full-sort reference tests still compare output distance bits, including the 8192x512 bench-scale digest.
- RNG unchanged: `knn_search` and the benchmark fixtures are deterministic.
- Golden outputs: run `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`.

Pre-score: Impact `2.0` x Confidence `0.75` / Effort `0.6` = `2.50`.

## Pass 4-6 candidate result

Proof:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-api knn_search -- --nocapture
```

RCH selected `vmi1167313`; all 3 focused KNN tests passed, including the bench-scale full-sort bit-exact proof.

Golden SHA:

```bash
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
```

Passed for all locally present artifacts.

Candidate rebench:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts1`

```text
knn_search/8192x512_k8  time: [12.614 ms 13.449 ms 14.508 ms]
```

Score: `Impact 0.64 x Confidence 0.95 / Effort 0.6 = 1.01`, below the `2.0` gate.

Verdict: rejected. The `KNN_QUERY_TILE=32` source hunk was removed. Next primitive should not repeat query-panel-width tuning; route to a deeper exact distance-panel memory layout, e.g. point SoA/staged coordinate panels or a batched top-k workspace that improves locality without changing scan/tie semantics.
