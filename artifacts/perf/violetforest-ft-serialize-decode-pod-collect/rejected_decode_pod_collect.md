# ft-serialize decode pod-collect rejection

Date: 2026-06-06
Agent: VioletForest
Scope: `ft-serialize`

## Target

Post-f32 reprofile made native f64 decode the dominant serializer panel row:

```text
native_state_dict/decode_many_small_f64_1024x4: [344.04 us 357.44 us 369.17 us]
```

## Candidate

Temporarily enabled bytemuck's `extern_crate_alloc` feature and replaced the
little-endian f64 payload decode loop with `bytemuck::pod_collect_to_vec`.
The intent was to test a safe-Rust typed byte collect that handles unaligned
input while avoiding per-value scalar byte assembly.

## Proof

Golden checksum:

```text
artifacts/optimization/golden_outputs/ft_serialize_decode_pass19.txt: OK
```

Focused proof test:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-serialize native_decode_many_small_f64_golden_summary_matches_fixture -- --nocapture
```

Result: passed.

Isomorphism:

Ordering: entry order, key parsing, shape parsing, dtype dispatch, and
`BTreeMap` insertion order were unchanged.

Tie-breaking: no comparator or tie-breaking path changed.

Floating point: no arithmetic changed. The candidate copied the exact f64
payload bytes into an aligned typed `Vec<f64>` on little-endian targets; the
big-endian path retained scalar `from_le_bytes`.

RNG: no RNG state or sampling path is involved.

## Benchmark

Baseline from same serializer panel:

```text
vmi1227854 native_state_dict/decode_many_small_f64_1024x4: [344.04 us 357.44 us 369.17 us]
```

Candidate command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

RCH selected `ts1`, not the requested baseline worker:

```text
ts1 native_state_dict/decode_many_small_f64_1024x4: [350.81 us 354.68 us 358.71 us]
```

The observed cross-worker delta is only `357.44 us -> 354.68 us` p50, about
0.8 percent. That does not clear the Score >= 2.0 keep gate.

Score: Impact 1 * Confidence 1 / Effort 1 = 1.0, reject.

## Decision

Rejected. Source was reverted to the committed width-4 f64 special-case reader.
No behavior change remains.

Next deeper target: replace the many-small-tensor decode structure rather than
the f64 scalar conversion loop. The next attempt should attack per-entry
metadata work, likely via a dedicated small-tensor native decode path that
precomputes fixed-width tensor layout and reduces repeated key/shape/map work,
with a target ratio of at least 2x on `decode_many_small_f64_1024x4`.
