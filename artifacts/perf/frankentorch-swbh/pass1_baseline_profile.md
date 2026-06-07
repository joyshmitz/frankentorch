# frankentorch-swbh pass 1 baseline/profile

Date: 2026-06-07

Skill: `extreme-software-optimization`

Scope: evidence only. No source edits were made in this pass.

## Target

- Bead: `frankentorch-swbh`
- Crate: `ft-api`
- Criterion row: `knn_search/8192x512_k8`
- Initial main `HEAD` observed before scratch creation: `e84252c5`
- Current main `HEAD` after the pass: `84555b8e perf(ft-api): reject swbh knn threshold pruning`

The main worktree initially had an uncommitted candidate hunk in
`crates/ft-api/src/lib.rs`: partial-distance early pruning in
`FrankenTorchSession::knn_search`. During this pass another agent landed
`84555b8e`, closed `frankentorch-swbh` as a rejected lever, and removed the
source hunk. This report keeps the baseline/after evidence collected here and
records the concurrent closeout rather than treating the tree as static.

## Baseline

Detached scratch worktree:

```text
/data/projects/.scratch/frankentorch-swbh-pass1-baseline-20260607T040934Z
```

Scratch command:

```bash
git worktree add --detach /data/projects/.scratch/frankentorch-swbh-pass1-baseline-20260607T040934Z HEAD
```

Scratch `HEAD`:

```text
e84252c5 chore(beads): close 57nj non-bit-exact conv2d backward lever
```

Benchmark command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Remote worker:

```text
ts1 at ubuntu@192.168.1.107
```

Criterion result:

```text
knn_search/8192x512_k8  time:   [8.3422 ms 8.4719 ms 8.6545 ms]
                        thrpt:  [484.64 Melem/s 495.09 Melem/s 502.78 Melem/s]
```

## Candidate after run

Command in `/data/projects/frankentorch`:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Remote worker:

```text
ts1 at ubuntu@192.168.1.107
```

Criterion result:

```text
knn_search/8192x512_k8  time:   [8.0952 ms 8.1730 ms 8.2532 ms]
                        thrpt:  [508.21 Melem/s 513.19 Melem/s 518.12 Melem/s]
Found 1 outliers among 20 measurements (5.00%)
  1 (5.00%) high mild
```

Median delta from this pass's detached baseline:

```text
8.4719 ms -> 8.1730 ms = 1.0366x faster, about 3.53% lower median time
```

This is not sufficient evidence to keep the lever because the same bead was
closed concurrently by another agent with a same-worker rejection artifact:
`artifacts/perf/frankentorch-ruby-knn-distance-panel/report.md` reports
`7.4241 ms -> 7.7303 ms` on `vmi1293453` and the source hunk was removed.

## Proof and golden status

Focused proof command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-api knn_search -- --nocapture
```

Result:

```text
running 3 tests
test tests::knn_search_zero_k_returns_empty_outputs ... ok
test tests::knn_search_streaming_topk_matches_full_sort_reference_bit_exact ... ok
test tests::knn_search_bench_scale_matches_full_sort_reference_bit_exact ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 1923 filtered out; finished in 2.40s
```

The focused command emitted three pre-existing unrelated `unused_mut` warnings
in later `ft-api` tests, but the KNN proof itself passed and the command exited
0.

Golden checksum command:

```bash
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
```

Result: passed for all locally present entries, including:

```text
artifacts/optimization/golden_outputs/ft_api_knn_topk_frankentorch-kgs4-34.txt: OK
artifacts/optimization/golden_outputs/ft_api_knn_search_frankentorch-66a9-pass2.txt: OK
artifacts/optimization/golden_outputs/ft_api_knn_search_frankentorch-66a9-pass5.txt: OK
```

KNN digest constants exercised by
`knn_search_bench_scale_matches_full_sort_reference_bit_exact`:

```text
idx_acc  = 0x31c1_945c_1c1b_dd25
dist_acc = 0x5146_b549_cd8f_131d
```

## Isomorphism status

- Ordering preserved: yes for the candidate design. Query tile order, local
  query order, and ascending point scan order are unchanged.
- Tie-breaking unchanged: yes. The prune gate uses strict `<` against the
  current worst retained squared distance, matching the existing top-k helper's
  equal-distance behavior.
- Floating point: retained candidates still use the same squared-distance terms
  and final `sqrt`. Skipped candidates are only those whose partial squared
  distance cannot enter the full top-k set.
- RNG: none in `knn_search`; the benchmark and proof fixtures are deterministic.
- Golden outputs: focused KNN digest proof passed; repository golden checksum
  verification passed for locally present files.

## Cost evidence and residual target

Deterministic fixture size:

```text
8192 points * 512 queries = 4,194,304 point/query pairs
k = 8
query tile = 16
```

Scalar partial-threshold model over the benchmark fixture:

```text
pairs=4194304
full_distance_z_computed=139378 (3.323%)
dx_pruned=3980887 (94.912%)
xy_pruned=74039 (1.765%)
consider_calls=139378 (3.323%)
```

Interpretation:

- The scalar threshold idea can skip most `dy/z` and insertion-helper work after
  top-k buffers fill, but it still executes a branch-heavy scalar `dx^2` check
  for every pair.
- The pass-local `ts1` run showed only a small 3.53% median improvement, while
  the concurrent closeout's same-worker artifact showed a regression and removed
  the hunk.
- The residual target should not be another scalar partial-prune tweak. The next
  profile-backed primitive should be materially different: an exact
  cache-tiled/SoA distance panel with portable SIMD and a k=8 register/top-k
  update path, or a re-profiled non-KNN hotspot if the closed bead leaves no
  open KNN work.

## Current bead state

`br show frankentorch-swbh --json` after the pass reports `closed`.

Close reason in the tracker:

```text
Rejected: exact KNN partial-distance threshold pruning preserved full-sort/golden KNN output, but same-worker vmi1293453 Criterion regressed knn_search/8192x512_k8 from 7.4241 ms to 7.7303 ms median (0.960x). Source hunk removed; artifact: artifacts/perf/frankentorch-ruby-knn-distance-panel/report.md.
```

`br ready --json` after the pass: `[]`.

## Files changed by this pass

- `artifacts/perf/frankentorch-swbh/pass1_baseline_profile.md`

No source files were edited. No files were deleted. The detached scratch
worktree was intentionally left in place.
