# frankentorch-dw9y pass 14 rejection: borrowed UTF-8 key materialization

## Target

- Bead: `frankentorch-dw9y`
- Benchmark: `native_state_dict/decode_many_small_f64_1024x4`
- Crate: `ft-serialize`
- Worker: `ts1`
- Candidate lever: replace `String::from_utf8(slice.to_vec())` with borrowed `std::str::from_utf8(slice)?.to_owned()` while leaving the returned `BTreeMap<String, DenseTensor>` contract unchanged.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1293453 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

RCH selected worker `ts1`.

Result:

```text
native_state_dict/decode_many_small_f64_1024x4
time: [515.40 us 529.54 us 545.17 us]
```

## Candidate Proof

The candidate was run from scratch mirror `/data/projects/.scratch/frankentorch-dw9y-after-rch-20260606T032443Z` so RCH artifact retrieval could not rewrite the live checkout.

Focused proofs passed:

- `cargo test -p ft-serialize load_state_dict_rejects`
- `cargo test -p ft-serialize native_decode_many_small_f64_golden_summary_matches_fixture`
- `cargo test -p ft-serialize native_format_width4_f64_preserves_raw_bits`
- `cargo test -p ft-serialize native_format_rejects_truncated_width4_f64_data`
- `ubs .skill-loop-progress.md artifacts/perf/frankentorch-ftserialize-f64decode/rejected_borrowed_utf8_key.md`

Isomorphism ledger:

- Ordering: unchanged; output is still `BTreeMap<String, DenseTensor>` and insertion still uses `result.entry(key)`.
- Duplicate-key tie behavior: unchanged; duplicate detection still happens through `BTreeMap::entry` with the same key bytes materialized as the same owned `String`.
- Floating point: unchanged; no payload decode, shape, dtype, or arithmetic path changed.
- RNG: not involved.
- Error behavior: candidate preserved the same `"invalid UTF-8 in key"`, duplicate-key, truncation, raw-bit, and golden-summary proof paths.

## After

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Result:

```text
native_state_dict/decode_many_small_f64_1024x4
time: [530.05 us 539.33 us 547.22 us]
```

## Confirmation Pair

Because an additional live-worktree log (`after_key_from_utf8_slice.txt`) reported a contradictory fast interval while RCH was retrieving artifacts from `/data/projects/frankentorch`, that log is treated as invalid supplemental evidence. The decision uses scratch-controlled clean and candidate mirrors only.

Clean baseline confirmation:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 8 --sample-size 30
native_state_dict/decode_many_small_f64_1024x4
time: [352.13 us 362.84 us 376.99 us]
```

Candidate confirmation:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 8 --sample-size 30
native_state_dict/decode_many_small_f64_1024x4
time: [384.02 us 397.57 us 410.64 us]
```

## Verdict

Rejected and source hunk removed.

Same-worker median regressed in both scratch-controlled comparisons:

- Short run: `529.54 us -> 539.33 us` (`0.982x`, about `1.85%` slower)
- Confirmation run: `362.84 us -> 397.57 us` (`0.913x`, about `9.57%` slower)

Score is below the keep gate:

```text
Impact 0.0 x Confidence 4.0 / Effort 1.0 = 0.0
```

## Next Primitive

Do not retry key-materialization micro-levers. The next attack should be a structural native decode primitive from the alien-graveyard fixed-width/segmented-data family: parse tensor records into a compact record index over borrowed byte ranges, decode fixed-width f64 payloads in a batched/segmented pass, and materialize owned `String` keys only at the final map boundary. Target ratio: at least `1.25x` on `decode_many_small_f64_1024x4` while preserving the same map ordering, duplicate-key failure point, exact f64 raw bits, and all malformed-input diagnostics.
