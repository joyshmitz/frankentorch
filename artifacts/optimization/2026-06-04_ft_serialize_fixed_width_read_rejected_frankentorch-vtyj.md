# ft-serialize Fixed-Width Header Read Rejection

- Bead: `frankentorch-vtyj`
- Crate: `ft-serialize`
- Target: `native_state_dict/decode_many_small_f64_1024x4`
- Skills: `/repeatedly-apply-skill`, `/extreme-software-optimization`, `/alien-graveyard`, `/alien-artifact-coding`

## Profile Target

The native FTSV decoder reads many small tensor records. The profile-backed
surface for this bead was the fixed-width header path:

```text
read_u32/read_u64 -> read_fixed_bytes::<4/8>
```

Baseline recorded in the bead on `vmi1149989`:

```text
native_state_dict/decode_many_small_f64_1024x4: [246.90 us 262.42 us 285.89 us]
```

## Attempted Lever

Replace the post-bounds-check `bytes_to_array(...).try_into()` conversion in
`read_fixed_bytes` with a stack array plus `copy_from_slice`:

```text
let mut bytes = [0u8; N];
bytes.copy_from_slice(&data[*pos..end]);
```

This was intentionally narrower than earlier rejected payload-loop decode
attempts: it only touched 4-byte and 8-byte header reads after the exact bounds
check had already succeeded.

## Isomorphism

The candidate was behavior-preserving but slower:

- Ordering preserved: header reads, tensor key parsing, dtype parsing, payload
  parsing, and `BTreeMap` insertion order were unchanged.
- Tie-breaking unchanged: no comparator or tie-breaking path is involved.
- Floating-point unchanged: f64 payload bytes and `from_le_bytes` conversion
  stayed on the existing path.
- RNG unchanged: native decode uses no RNG.
- Error behavior unchanged: overflow and truncation checks still happen before
  the byte copy, with the same `TensorIOError::Corrupt` reason text.
- Golden output: existing native decode fixture remains
  `93332e6e21332f43535b64ab5fe3224f22213becb95cb4d3b6e0ed9888dbe943`.

## Benchmark Result

Same-worker candidate run:

```text
worker: vmi1149989
command: RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 5 --sample-size 20
native_state_dict/decode_many_small_f64_1024x4: [528.33 us 557.76 us 588.77 us]
```

Result: p50 regressed from `262.42 us` to `557.76 us` on the same worker.

Score: Impact 0 x Confidence 4 / Effort 1 = 0.0.

## Closeout

Rejected. The source hunk was reverted; no runtime change remains. The next
serialization pass should not keep tightening this header-read micro-path.
Pivot to a different primitive, such as a broader zero-copy framing/layout
rewrite or move to a currently higher-scored profile-backed non-serialization
hotspot.
