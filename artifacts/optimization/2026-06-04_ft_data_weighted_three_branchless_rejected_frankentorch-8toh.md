# ft-data three-weight sampler branchless threshold rejected (frankentorch-8toh)

## Target

- Skill loop: `/repeatedly-apply-skill` applying `/extreme-software-optimization`, with `/alien-graveyard` and `/alien-artifact-coding` discipline for structural pivots.
- Bead: `frankentorch-8toh`
- Crate: `ft-data`
- Benchmark: `sampler/weighted_three_positive_3x1m`
- Profile-backed source target: `WeightedRandomSampler::indices` three-weight fast path loops over 1,000,000 samples with an if/else threshold chain. The four-weight path already uses a branchless threshold helper.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-data --bench sampler_bench -- sampler/weighted_three_positive_3x1m --warm-up-time 1 --measurement-time 10 --sample-size 20
```

Worker/result:

```text
ts1: [4.5017 ms 4.5622 ms 4.6206 ms]
```

Profile-time command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-data --bench sampler_bench -- sampler/weighted_three_positive_3x1m --profile-time 10
```

Result: completed on `ts1`; Criterion did not emit function attribution. The profiled row maps directly to `WeightedRandomSampler::indices` for three weights.

## Lever Attempted

Add a `threshold_index_3` helper:

```text
usize::from(u > first_threshold) + usize::from(u > second_threshold)
```

and use it only in the three-weight fast path.

## Isomorphism proof

- Validation and error ordering were unchanged: `num_samples`, empty weights, finite/non-negative weights, finite sum, positive sum, and threshold extraction still happened in the same order.
- Cumulative threshold arithmetic and normalization were unchanged.
- RNG call count and order were unchanged: exactly one `next_u64` per requested sample, with the same conversion to uniform f64.
- Boundary semantics were unchanged: the old `u <= threshold` branches are equivalent to incrementing only when `u > threshold`.
- Output order, tie behavior, and non-RNG floating-point operations were unchanged.
- Golden fixture checksum remained unchanged:
  - `ft_data_weighted_sampler_branchless_frankentorch-sx6l.txt`: `5139c7e647c38c1f9000e022197f10be6426544e9561ea724e0c60fa08062d90`

## Validation

```text
cargo fmt -p ft-data -- --check
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-data weighted -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-data --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-data --all-targets --no-deps -- -D warnings
sha256sum artifacts/optimization/golden_outputs/ft_data_weighted_sampler_branchless_frankentorch-sx6l.txt
```

Result: pass. The remote check/clippy output included unrelated dependency warnings from active peer surfaces; `ft-data` passed.

## Re-benchmark

RCH did not honor the worker hint for the after benchmark attempts:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-data --bench sampler_bench -- sampler/weighted_three_positive_3x1m --warm-up-time 1 --measurement-time 10 --sample-size 20
RCH_REQUIRE_REMOTE=1 RCH_WORKERS=ts1 rch exec -- cargo bench -p ft-data --bench sampler_bench -- sampler/weighted_three_positive_3x1m --warm-up-time 1 --measurement-time 10 --sample-size 20
```

Both after runs landed on `ts2`:

```text
ts2: [6.9703 ms 7.0019 ms 7.0257 ms]
ts2: [6.9876 ms 7.0427 ms 7.0916 ms]
```

These are not comparable to the `ts1` baseline and are also a negative cross-worker signal.

## Score

```text
Impact 0 x Confidence 4 / Effort 1 = 0.0
```

Rejected: the source hunk was removed. No runtime change is kept.

## Pivot

Stop tuning the small-cardinality branch chain. The next ft-data sampler attack should use a structurally different primitive only when profile-backed, such as a Walker/Vose alias table for large-cardinality weighted sampling, or a different benchmark row with a clear allocation or RNG hotspot.
