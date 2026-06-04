# ft-data four-weight sampler threshold classification

Bead: `frankentorch-sx6l`

## Target

`WeightedRandomSampler::indices` has dedicated fast paths for tiny positive
weight vectors. RCH Criterion baseline on `ts2` showed the four-weight path as
the useful target in the tiny weighted sampler group:

```text
weighted_two_positive_2x1m   [6.4974 ms 6.5346 ms 6.5681 ms]
weighted_three_positive_3x1m [6.8145 ms 6.8537 ms 6.8794 ms]
weighted_four_positive_4x1m  [11.473 ms 11.554 ms 11.602 ms]
```

## Lever

Replace the four-weight nested branch chain with a threshold-count helper:

```text
index = (u > t0) + (u > t1) + (u > t2)
```

The first broad attempt applied the same pattern to two- and three-weight fast
paths. Same-worker rebench rejected that wider surface because the two- and
three-weight rows regressed. The retained one-lever commit only changes the
four-weight path, where the branchless threshold count produced a stable win.

Alien primitive: branchless decision table / threshold count for tiny categorical
sampling. It changes the classification primitive, not the RNG stream or
validation path.

## Benchmark

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo bench -p ft-data --bench sampler_bench -- weighted --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Rejected broad attempt on `ts2`:

```text
weighted_two_positive_2x1m   [6.5877 ms 6.6906 ms 6.8159 ms]
weighted_three_positive_3x1m [6.9433 ms 6.9811 ms 7.0078 ms]
weighted_four_positive_4x1m  [8.2650 ms 8.3422 ms 8.3845 ms]
```

Retained narrowed lever on `ts2`:

```text
weighted_single_positive_1x1m [91.193 us 92.547 us 94.298 us]
weighted_two_positive_2x1m    [6.5204 ms 6.7691 ms 7.0329 ms]
weighted_three_positive_3x1m  [6.7519 ms 6.8338 ms 6.8827 ms]
weighted_four_positive_4x1m   [8.2732 ms 8.3502 ms 8.3904 ms]
```

Four-weight p50 speedup: `11.554 / 8.3502 = 1.383x`.

Score: `Impact 3 * Confidence 4 / Effort 1 = 12.0`.

## Isomorphism

- Ordering preserved: each sample still performs one RNG draw and appends one
  index in the same loop order.
- Tie behavior preserved: the old chain selected the lower bucket on exact
  threshold equality (`u <= threshold`). The helper uses strict `u > threshold`
  counts, so exact equality still maps to the lower bucket.
- Floating point preserved: thresholds and `u` are computed identically; only
  the comparisons are reorganized.
- RNG preserved: seed, `next_u64` call count, right shift, and `[0, 1)` scaling
  are unchanged.
- Validation/error order preserved: all weight validation and cumulative
  threshold construction happen before the changed classification loop.

## Golden

Fixture:

```text
artifacts/optimization/golden_outputs/ft_data_weighted_sampler_branchless_frankentorch-sx6l.txt
```

SHA256:

```text
5139c7e647c38c1f9000e022197f10be6426544e9561ea724e0c60fa08062d90
```

## Validation

Passed:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-data weighted_sampler -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-data --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-data --all-targets --no-deps -- -D warnings
cargo fmt -p ft-data --check
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
git diff --check -- crates/ft-data/src/lib.rs artifacts/optimization/golden_checksums.txt artifacts/optimization/golden_outputs/ft_data_weighted_sampler_branchless_frankentorch-sx6l.txt .beads/issues.jsonl
```

The broader `cargo clippy -p ft-data --all-targets -- -D warnings` command
failed in unrelated workspace dependency `ft-api` before reaching this crate;
the dependency-suppressed ft-data clippy gate above passed.

UBS was run on the changed files:

```bash
ubs crates/ft-data/src/lib.rs artifacts/optimization/2026-06-04_ft_data_branchless_weighted_sampler_frankentorch-sx6l.md artifacts/optimization/golden_outputs/ft_data_weighted_sampler_branchless_frankentorch-sx6l.txt artifacts/optimization/golden_checksums.txt .skill-loop-progress.md .beads/issues.jsonl
```

It exited `1` because its file-wide heuristics report pre-existing `ft-data`
findings, including a false-positive secret-comparison classification on a
shape/channel check and existing test panic-surface inventories. UBS's embedded
`cargo fmt`, `cargo clippy`, `cargo check`, test-build, audit, and deny checks
all passed.
