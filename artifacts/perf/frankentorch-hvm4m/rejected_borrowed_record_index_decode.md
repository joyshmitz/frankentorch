# frankentorch-hvm4m rejection: borrowed record-index native decode

Date: 2026-06-14
Agent: IvoryDeer
Scope: `ft-serialize`

## Target

Profile-backed row:

```text
cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

This is the same many-small native f64 decode row left after prior rejected
key-materialization, payload-collect, sorted-vector, and rank-1 metadata
micro-levers. The candidate therefore attacked the parser structure: borrowed
record indexing and delayed materialization.

## Baseline

Primary same-worker baseline from the clean scratch mirror:

```text
RCH worker: vmi1293453
native_state_dict/decode_many_small_f64_1024x4
time: [314.05 us 318.21 us 322.09 us]
```

The earlier live-worktree baseline on `vmi1167313` was:

```text
time: [800.51 us 846.11 us 906.43 us]
```

That worker was not used for the keep/reject decision because the accepted
comparison pair must be same-worker.

## Candidate

Temporary source hunk, now removed:

- Parse each native state-dict entry into `NativeTensorRecord<'a>` over borrowed
  key and payload ranges.
- Validate sorted unique keys with a fast adjacent comparison and build a
  borrowed `BTreeSet` only after the first out-of-order key.
- Decode payloads and allocate final `String` keys only after full record-index
  validation.

No format, dtype, tensor arithmetic, or public return type changed.

## Behavior Proof

Candidate-only focused proof:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1293453 RCH_PREFERRED_WORKER=vmi1293453 \
  rch exec -- cargo test -j 1 -p ft-serialize native_ -- --nocapture
```

Result: `15 passed; 0 failed`, including:

- `native_decode_many_small_f64_golden_summary_matches_fixture`
- `native_format_width4_f64_preserves_raw_bits`
- native truncation, shape overflow, excessive count, and round-trip tests

Additional malformed-input proof:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1293453 RCH_PREFERRED_WORKER=vmi1293453 \
  rch exec -- cargo test -j 1 -p ft-serialize load_state_dict_rejects -- --nocapture
```

Result: `3 passed; 0 failed`, covering duplicate keys, trailing bytes, and
impossible tensor counts.

Golden SHA-256 proof:

```text
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
```

Result: all present fixtures passed, including
`artifacts/optimization/golden_outputs/ft_serialize_decode_pass19.txt`.

Isomorphism obligations:

- Ordering: public output remained `BTreeMap<String, DenseTensor>`, preserving
  sorted key iteration.
- Duplicate/tie behavior: adjacent duplicate keys failed at the same point;
  out-of-order duplicate keys failed during the borrowed set validation before
  final materialization.
- Floating point: f64 payload conversion stayed `from_le_bytes`; raw-bit proof
  passed.
- RNG: no RNG state or sampling path is involved.
- Diagnostics: invalid UTF-8, truncation, unsupported dtype, shape overflow,
  duplicate key, and trailing-byte contracts were preserved by focused tests.

## Rebenchmark

Same-worker candidate run:

```text
RCH worker: vmi1293453
native_state_dict/decode_many_small_f64_1024x4
time: [332.15 us 339.27 us 346.14 us]
```

Median delta:

```text
318.21 us -> 339.27 us
speedup: 0.938x
```

## Decision

Rejected. The candidate regressed the profile-backed row, so the source hunk was
removed. Final `crates/ft-serialize/src/lib.rs` has no diff for this bead.

Score:

```text
Impact 0.0 x Confidence 5.0 / Effort 2.0 = 0.0
```

## Next Primitive

Do not retry key materialization, sorted-map construction, delayed materialization,
rank-1 metadata, or scalar payload collect micro-levers.

Next attack should be a different primitive: an owned batch/slab native decode
path for many small tensors, likely requiring an `ft-core` storage representation
that can safely share a decoded f64 slab across per-key tensor views while
preserving the public `BTreeMap<String, DenseTensor>` order and exact raw bits.
Target ratio: at least `2.0x` on `decode_many_small_f64_1024x4`.
