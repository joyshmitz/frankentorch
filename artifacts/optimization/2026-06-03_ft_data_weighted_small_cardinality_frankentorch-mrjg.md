# ft-data WeightedRandomSampler three-weight threshold cascade

Bead: `frankentorch-mrjg`
Agent: TurquoisePine
Date: 2026-06-03

## Target

Profile-backed fallback target selected after `br ready --json` returned no ready perf bead and active perf/no-gaps beads were already claimed on `ft-kernel-cpu` and `ft-optim`.

Benchmark target:

```bash
rch exec -- cargo bench -p ft-data --bench sampler_bench -- sampler/weighted_three_positive_3x1m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Workload:

- `WeightedRandomSampler::new(vec![1.0, 3.0, 6.0], 1_000_000).with_seed(0x5151_0003)`
- Existing hot path built normalized cumulative thresholds and used `binary_search_by(total_cmp)` for every sample.

Alien primitive:

- `alien_cs_graveyard.md` section 16.7, "The Constants Kill You": benchmark small domains before accepting generic asymptotic machinery.
- FrankenSuite summary mitigation: "Profile-first; only use where constants win."
- Applied as a fixed small-cardinality threshold cascade for exactly three buckets.

## Baseline

Baseline command:

```text
rch exec -- env CARGO_TARGET_DIR=target/turquoise-pine-ft-data-mrjg-baseline cargo bench -p ft-data --bench sampler_bench -- sampler/weighted_three_positive_3x1m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `vmi1156319`

```text
time: [11.431 ms 11.764 ms 12.144 ms]
```

The baseline kept the benchmark target and golden-order test present, but
temporarily restored the old source path with no three-weight fast path.

## Lever

One lever only: add a `self.weights.len() == 3` branch after all validation and after the existing one- and two-weight paths.

The branch computes:

- `first_threshold = cumulative[0] / total`
- `second_threshold = cumulative[1] / total`

It then uses the same `SimpleRng::new(self.seed)` and same uniform draw expression:

```rust
(rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
```

Mapping:

- `u <= first_threshold` -> 0
- `u <= second_threshold` -> 1
- otherwise -> 2

## Proof

Golden output:

- `artifacts/optimization/golden_outputs/ft_data_weighted_small_cardinality_frankentorch-mrjg.txt`
- sha256 `72d4f7b0e3979bcde76daa2a06f0cbe27d81189b26e674817762e05daf463d84`

Verification:

```bash
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
rch exec -- env CARGO_TARGET_DIR=target/turquoise-pine-ft-data-mrjg-verify cargo test -p ft-data weighted_random_sampler -- --nocapture
```

Results:

- `weighted_random_sampler`: 14 passed, 0 failed on `vmi1227854`.
- Checksum ledger: all present golden outputs OK, including `ft_data_weighted_small_cardinality_frankentorch-mrjg.txt`.

Isomorphism:

- Ordering preserved: yes. One output index is pushed per RNG draw in the same loop order.
- Tie-breaking preserved: yes. `binary_search_by(total_cmp)` returns the lower bucket on exact threshold equality; the cascade uses `<=` for each threshold.
- Floating-point preserved: yes for len 3. Weight validation and cumulative addition occur in the same order before the branch; thresholds are the same `cumulative[i] / total` values used by the old normalized cumulative vector.
- RNG preserved: yes. Seed, RNG state transition, draw count, and uniform f64 conversion are unchanged.
- Error classes preserved: yes. The branch runs only after `num_samples`, non-empty weights, finite/non-negative weights, finite sum, and positive sum validation.

## After

After command:

```text
rch exec -- env CARGO_TARGET_DIR=target/turquoise-pine-ft-data-mrjg-after cargo bench -p ft-data --bench sampler_bench -- sampler/weighted_three_positive_3x1m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `vmi1153651`

```text
time: [7.1255 ms 7.4022 ms 7.6631 ms]
```

Delta:

- Mean estimate: 11.764 ms -> 7.4022 ms
- Elapsed-time improvement: 37.1%
- Throughput multiplier: 1.59x
- `rch exec` did not expose direct worker pinning, so this is a cross-worker
  comparison with confidence penalized despite the large effect size.

Score:

- Impact 3 x Confidence 2 / Effort 1 = 6.0
- Verdict: keep.

## Gates

Passed:

```bash
rch exec -- env CARGO_TARGET_DIR=target/turquoise-pine-ft-data-mrjg-verify cargo check -p ft-data --all-targets
rch exec -- env CARGO_TARGET_DIR=target/turquoise-pine-ft-data-mrjg-verify cargo clippy -p ft-data --all-targets --no-deps -- -D warnings
cargo fmt -p ft-data --check
git diff --check
```

`rch exec -- cargo fmt -p ft-data --check` classified rustfmt as a
non-compilation command and returned immediately, so rustfmt was run directly
and package-scoped.

UBS:

```bash
ubs crates/ft-data/src/lib.rs crates/ft-data/benches/sampler_bench.rs artifacts/optimization/2026-06-03_ft_data_weighted_small_cardinality_frankentorch-mrjg.md artifacts/optimization/golden_checksums.txt artifacts/optimization/golden_outputs/ft_data_weighted_small_cardinality_frankentorch-mrjg.txt
```

UBS exited nonzero on existing broad `ft-data` inventory:

- 2 critical false-positive/legacy secret-compare findings in unrelated shape/test comparisons.
- 260 warnings and 47 info items across the existing file.
- Built-in UBS fmt, clippy, cargo check, test-build, cargo-audit, and cargo-deny probes passed.
- No critical finding is from the new sampler len-3 fast path.
