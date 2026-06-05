# ft-data large-cardinality weighted sampler

Bead: frankentorch-j54u
Date: 2026-06-05
Agent: BoldOx

## Target

`WeightedRandomSampler::indices` for generic large-cardinality positive weights.

Benchmark:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo bench -p ft-data --bench sampler_bench weighted_4096_positive_4096x262k -- --warm-up-time 1 --measurement-time 5 --sample-size 20
```

## Baseline

Worker: ts2

```text
sampler/weighted_4096_positive_4096x262k
time: [10.033 ms 10.072 ms 10.102 ms]
```

Baseline used the same benchmark row with the hot comparator call restored to
`c.total_cmp(&u)`, leaving the bench row and golden fixture unchanged.

## Lever

Use a finite-only comparator for the normalized cumulative thresholds in the
existing binary search. Validation rejects NaN and infinity before threshold
construction, and the random sample is generated in `[0, 1)`, so the hot search
does not need the full NaN-aware `f64::total_cmp` path.

## Isomorphism Proof

- Ordering preserved: yes. Threshold vector construction and binary-search insertion handling are unchanged.
- Tie-breaking unchanged: yes. Equality still returns `Ok(i)`, preserving the existing lower-bucket behavior for an exact threshold hit.
- Floating-point: identical. Weight summation, normalization, RNG draw conversion, and sampled output values are unchanged.
- RNG seeds: unchanged. The seed and number of `next_u64` calls per sample are unchanged.
- Golden outputs: unchanged. The 32-sample deterministic fixture is pinned by
  sha256 `ff33133ed2d4dab1627878e7ba7f7d1fe4c426a064be2749cd75a740b210b3e8`.

## After

Worker: ts2

```text
sampler/weighted_4096_positive_4096x262k
time: [8.5344 ms 8.5718 ms 8.5982 ms]
```

Pinned confirmation on worker ts2:

```text
sampler/weighted_4096_positive_4096x262k
time: [8.6181 ms 8.6684 ms 8.7169 ms]
```

Delta by p50: `10.072 ms -> 8.6684 ms`, about `13.9%` faster (`1.16x`) on
the confirmation run.

Score: Impact `3` x Confidence `2` / Effort `1` = `6.0`, above the required
`2.0` threshold. Confidence is based on a same-worker RCH A/B plus the golden
fixture preserving sampled order.

## Verification

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-data weighted_sampler -- --nocapture`:
  passed 5 filtered tests, including the large-cardinality golden fixture and
  comparator equivalence check.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-data --all-targets`:
  passed. Dependency crates still emitted pre-existing warnings outside this
  bead.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-data --all-targets --no-deps -- -D warnings`:
  passed. Dependency crates still emitted pre-existing warnings outside this
  bead.
- `cargo fmt -p ft-data --check`: passed. RCH refused fmt as a non-compilation
  command when remote execution was required.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`:
  passed, including
  `artifacts/optimization/golden_outputs/ft_data_weighted_large_cardinality_frankentorch-j54u.txt`.
- `git diff --check` on bead-owned files: passed.
- `ubs` on bead-owned files: exit 0. UBS reported only existing inventory-style
  warnings in `ft-data`, with zero critical findings.
