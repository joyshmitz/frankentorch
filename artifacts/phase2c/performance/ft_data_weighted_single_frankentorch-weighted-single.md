# ft-data WeightedRandomSampler Single-Weight Fast Path

- Agent: TurquoisePine
- Thread: `perf-ft-data-weighted-single`
- Bead: pending; `.beads/issues.jsonl` was reserved by SilentSnow during this slice
- Crate: `ft-data`
- Target: `sampler/weighted_single_positive_1x1m`

## Baseline

Command:

```text
rch exec -- cargo bench -p ft-data --bench sampler_bench -- sampler/weighted_single_positive_1x1m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Result:

```text
worker: vmi1149989
time: [4.4825 ms 4.9919 ms 5.7296 ms]
```

## Lever

After preserving the existing `WeightedRandomSampler::indices` validation order,
return `vec![0; num_samples]` when there is exactly one valid positive weight.

The normal weighted sampling path for two or more weights is unchanged.

## Isomorphism Proof

- Ordering preserved: the old single-weight path always returned index `0` for every sample because the only cumulative threshold was `1.0` and generated `u` is in `[0, 1)`.
- Tie-breaking unchanged: no observable tie path exists for the single-weight case.
- Floating-point unchanged observably: validation still checks finite/non-negative/sum-positive first; skipped normalization would only produce the single threshold `1.0`.
- RNG unchanged observably: the RNG is local to `indices`; no caller-visible RNG state escapes.
- Error order preserved: zero samples, non-finite weight, and zero total still fail before the fast path.
- Golden output: `e080716a15c41195de5a93165178bb7a0e6a78d361a6c8e23b005c590b3c75d7`.

## Rebenchmark

Command:

```text
rch exec -- cargo bench -p ft-data --bench sampler_bench -- sampler/weighted_single_positive_1x1m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Result:

```text
worker: vmi1227854
time: [33.279 us 34.083 us 34.850 us]
```

Delta:

- p50: `4.9919 ms -> 34.083 us`
- speedup: about `146x`
- score: Impact `2` x Confidence `4` / Effort `1` = `8.0`
- decision: keep

## Gates

- `rch exec -- cargo test -p ft-data weighted_random_sampler -- --nocapture`: passed, 12 tests.
- `rch exec -- cargo check -p ft-data --all-targets`: passed.
- `rch exec -- cargo clippy -p ft-data --all-targets --no-deps -- -D warnings`: passed.
- `rch exec -- cargo fmt -p ft-data --check`: passed.
- `sha256sum -c artifacts/phase2c/performance/ft_data_weighted_single_frankentorch-weighted-single.sha256`: passed.
- `git diff --check`: passed.
- `ubs crates/ft-data/src/lib.rs crates/ft-data/benches/sampler_bench.rs ...`: nonzero from pre-existing ft-data inventory; UBS internal fmt/clippy/check/test-build subchecks passed.
