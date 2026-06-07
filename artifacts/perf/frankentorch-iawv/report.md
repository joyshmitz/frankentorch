# frankentorch-iawv KNN staged coordinate-panel rejection

## Target

Profile-backed target: `knn_search/8192x512_k8` in `ft-api`.

Rejected predecessor families:

- scalar partial-distance threshold pruning (`frankentorch-swbh`)
- query-panel width tuning (`frankentorch-bbgu`)
- point-major all-query top-k workspace (`frankentorch-jod6`)

This pass tested the next exact-KNN family: staged coordinate-panel / finite
hot-loop specialization while preserving the strict top-k contract.

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Concrete worker: `vmi1167313`.

```text
knn_search/8192x512_k8  time: [17.154 ms 17.713 ms 18.371 ms]
```

## Candidate

One measured candidate combined the finite-input comparator specialization with
the staged point-coordinate panel. Because the branch advanced concurrently, the
same-worker run is treated as a rejection of this adjacent KNN loop-layout
family, not as a keepable one-lever source change.

The intended isomorphism obligations were:

- Ordering: batch order, query tile order, per-query point order, and output
  write order preserved.
- Tie-breaking: strict less-than insertion preserved; equal distances keep the
  earlier point index.
- Floating point: exact squared-distance expression `dx * dx + dy * dy + dz * dz`
  and final `sqrt` path preserved.
- RNG: no RNG path in `knn_search`.
- Autograd: KNN remains a value-only output path with `requires_grad=false`.

## Proof

Focused KNN proof:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-api knn_search -- --nocapture
```

Result: all 3 focused KNN tests passed, including the bench-scale full-sort
reference digest.

Golden SHA-256:

```bash
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
```

Result: all locally present golden outputs passed.

## Rebench

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1167313 rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

```text
knn_search/8192x512_k8  time: [16.712 ms 17.214 ms 17.811 ms]
```

Median ratio: `17.713 / 17.214 = 1.029x`.

Score: `Impact 1.03 x Confidence 0.95 / Effort 0.6 = 1.63`, below the `2.0`
gate.

Verdict: rejected. The finite-comparator and point-panel source hunks were
removed from the final working tree; no KNN hot-path source code is retained.

## Final validation

- `cargo test -p ft-api knn_search -- --nocapture`: passed 3/3 on the final
  source state.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`:
  passed for all locally present outputs.
- `cargo check -p ft-api --all-targets`: passed through `rch`.
- `ubs crates/ft-api/src/lib.rs`: timed out after more than 4 minutes in an
  `ast-grep` pass before emitting findings; artifact captured.
- `cargo fmt -p ft-api --check`: failed on broad current `ft-api`
  formatting drift outside the KNN rejection hunk; artifact captured.
- `cargo clippy -p ft-api --all-targets -- -D warnings`: failed on broad
  current `ft-api` test-lint drift outside the KNN rejection hunk; artifact
  captured.

Next primitive: re-profile and route to a disjoint structural target. If KNN
remains the top profile-backed target, attack a kernel-bound exact-distance
backend with an explicit FMA-neutral SIMD proof before lane parallelism.
