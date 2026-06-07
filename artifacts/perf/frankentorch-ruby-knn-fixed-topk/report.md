# frankentorch-tgst KNN fixed-size top-k maintenance

## Target

- Bead: `frankentorch-tgst`
- Profile-backed benchmark: `knn_search/8192x512_k8`
- Rationale: after `br ready --json --no-auto-import` returned no ready `[perf]` beads and current in-progress perf beads were already claimed by other agents, this pass used the existing Criterion KNN target as an independent profile-backed hotspot.

## One lever

Replace each query tile's per-query `Vec<(usize, f64)>` insertion/pop top-k maintenance with fixed preallocated index and distance buffers plus a retained-length array.

No distance formula, output shape, output dtype, output ordering rule, autograd routing, or RNG behavior changed.

## Same-worker benchmark

Command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Initial baseline on `ts1` at parent `9e70e7d8`:

```text
knn_search/8192x512_k8  time: [9.2421 ms 9.3108 ms 9.3708 ms]
```

Initial after on `ts1`:

```text
knn_search/8192x512_k8  time: [7.7284 ms 7.8545 ms 8.0200 ms]
```

After rebasing over current `origin/main`, final parent baseline on `ts1` at `886b244a`:

```text
knn_search/8192x512_k8  time: [9.3652 ms 9.5135 ms 9.6878 ms]
```

Final rebased after on `ts1` at `e903a63f`:

```text
knn_search/8192x512_k8  time: [7.6360 ms 7.6896 ms 7.7349 ms]
```

Final median speedup: `9.5135 / 7.6896 = 1.237x`.

Score: `Impact 2.2 x Confidence 0.95 / Effort 0.8 = 2.61`, keep.

## Isomorphism proof

- Ordering/tie-breaking: the old path inserted a candidate only when `d < retained_distance`; the new helper uses the same strict `partial_cmp(... Less)` rule. Equal distances do not move ahead of earlier retained points, and an equal-to-worst candidate is still ignored when the buffer is full.
- Floating point: every candidate distance still evaluates the same three squared coordinate deltas in the same expression, and each retained output still applies `sqrt()` only after top-k selection.
- RNG: `knn_search` contains no RNG, and the benchmark/test fixtures are deterministic.
- Shape/dtype: output tensor shapes and f64 index/distance storage are unchanged for both unbatched and batched inputs.
- Autograd: this API path remains value-only KNN output generation; no gradient routing was added or removed.

Golden output:

- `cargo test -p ft-api knn_search -- --nocapture` passed the existing full-sort-reference bit-exact tests on `ts1`.
- `golden_knn_contract.txt` records the proof obligations; `evidence.sha256` records its checksum and the raw focused test log checksum.

## Gates

- PASS: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-api knn_search -- --nocapture`
- PASS: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-api --all-targets`
- PASS: `git diff --check`
- PASS: same-worker Criterion rebench on `ts1` both before and after rebasing onto current `origin/main`
- BLOCKED: `cargo fmt -p ft-api --check` reports broad pre-existing formatting drift outside this hunk.
- BLOCKED: `cargo clippy -p ft-api --all-targets -- -D warnings` reports broad pre-existing `ft-api` lint debt.
- BLOCKED: `ubs crates/ft-api/src/lib.rs` timed out after 300 seconds without findings; the git commit hook repeated the UBS scan, timed out on the large-file scan without findings, and documented the `UBS_SKIP=1` bypass.
- BLOCKED: `br sync --flush-only` refused to export the stale tracker DB because that export would drop peer issue `frankentorch-ib63`; no force export was used.

## Next primitive

Re-profile after this KNN keep. If KNN remains visible, the next deeper primitive is an algorithmic tiled/SIMD distance panel that computes multiple point-query distances per tile while preserving the same strict top-k insertion contract; otherwise route to the next ready `[perf]` bead.
