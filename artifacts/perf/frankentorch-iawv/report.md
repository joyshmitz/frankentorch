# frankentorch-iawv KNN staged coordinate-panel keep

## Target

Profile-backed target: `knn_search/8192x512_k8` in `ft-api`.

Rejected predecessor families:

- scalar partial-distance threshold pruning (`frankentorch-swbh`)
- query-panel width tuning (`frankentorch-bbgu`)
- point-major all-query top-k workspace (`frankentorch-jod6`)

This pass tested the next exact-KNN family: staged point-coordinate SoA panels
inside the existing query tile, while preserving the original `partial_cmp`
candidate comparator for every input.

## Baseline

Final clean baseline command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKERS=fmd rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Source: detached clean gate worktree at `60f5edd3` with no candidate hunk.

Concrete worker: `fmd` via `RCH_WORKERS=fmd`.

```text
knn_search/8192x512_k8  time: [6.4342 ms 6.4421 ms 6.4520 ms]
```

## Candidate

One lever: `KNN_POINT_PANEL=64` stages point `x/y/z` coordinates into three
stack arrays, then scans each staged panel for each query in the current
`KNN_QUERY_TILE=16` query tile.

Top-k workspace is unchanged: `KNN_QUERY_TILE * k` indices and distances.

## Isomorphism proof

- Ordering: each query still sees points in ascending `pi` order across panels.
- Tie-breaking: the existing `partial_cmp(...) == Some(Less)` insertion
  comparator is unchanged; equal distances keep the earlier point index.
- Floating point: every candidate distance remains exactly
  `dx * dx + dy * dy + dz * dz`; output distances still take one final `sqrt`.
- RNG: `knn_search` remains deterministic and does not consume session RNG.
- Output contract: focused KNN tests compare against independent full-sort
  references and bench-scale golden output-bit digests.

Proof commands:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-api knn_search -- --nocapture
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
```

Results:

- Focused KNN proof passed 3/3 on `vmi1293453`, including bench-scale digest.
- Golden sha256 verification passed for all locally present outputs.
- `cargo check -p ft-api --all-targets` passed on `fmd`.
- `cargo fmt --check -p ft-api` failed on broad existing `ft-api`
  formatting drift outside the KNN panel hunk.
- `cargo clippy -p ft-api --all-targets -- -D warnings` failed on broad
  existing `ft-api` lint drift outside the KNN panel hunk.
- `ubs crates/ft-api/src/lib.rs` timed out after 300 seconds without emitting
  findings.

## Rebench

Final same-worker candidate command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKERS=fmd rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Concrete worker: `fmd` via `RCH_WORKERS=fmd`.

```text
knn_search/8192x512_k8  time: [4.3276 ms 4.3452 ms 4.3737 ms]
```

Median ratio: `6.4421 / 4.3452 = 1.483x`.

Score: `Impact 1.48 x Confidence 0.97 / Effort 0.70 = 2.05`.

Verdict: kept. The point-coordinate panel source hunk is retained.

## Notes

Earlier cross-worker, stale-HEAD, and combined finite-comparator experiments
were not used for the keep decision. The final gate is the matched `fmd`
clean-baseline / panel-only candidate pair above.
