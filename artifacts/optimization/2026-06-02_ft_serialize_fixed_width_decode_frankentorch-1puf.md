# ft-serialize Fixed-Width F64 Decode Rejection

- Bead: `frankentorch-1puf`
- Crate: `ft-serialize`
- Target: `native_state_dict/decode_many_small_f64_1024x4`
- Skills: `/extreme-software-optimization`, `/alien-graveyard`, `/alien-artifact-coding`

## Profile Target

`load_state_dict_from_bytes` decodes 1024 small f64 tensors in the native FTSV
state-dict format. The profiled candidate target was `read_f64_payload`, which
turns fixed-width little-endian byte chunks into `f64` values.

Baseline from the bead:

```text
worker: vmi1293453
command: rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
native_state_dict/decode_many_small_f64_1024x4: [342.17 us 345.59 us 349.52 us]
```

Rejected candidate:

```text
lever: replace manual Vec::with_capacity + push loop with chunks_exact().map(...).collect()
worker: vmi1227854
native_state_dict/decode_many_small_f64_1024x4: 411.56 us p50
```

Confirmation run while the draft hunk was still present:

```text
worker: vmi1156319
command: rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 5 --sample-size 20
native_state_dict/decode_many_small_f64_1024x4: [693.42 us 715.92 us 738.89 us]
```

The candidate failed the keep rule. Source was restored to the manual
preallocated push loop, so no runtime optimization shipped.

## Alien Recommendation Card

Candidate primitive: constants-kill-you fixed-width decode tightening through
array-sized chunk iteration.

Expected value after measurement: Impact 0 x Confidence 4 / Effort 1 = 0.

Fallback: keep the existing manual `Vec::with_capacity(numel)` plus
`chunks_exact(8)` push loop. It already preserves exact f64 bit patterns and is
faster than the tested iterator/collect variant.

## Isomorphism Proof

The rejected candidate was behavior-isomorphic but slower:

- Ordering: tensor-key parsing, BTreeMap insertion order, and duplicate-key
  rejection were unchanged.
- Tie-breaking: no comparator or tie-break behavior is involved.
- Floating point: each value used the same eight little-endian bytes and the
  same `f64::from_le_bytes` conversion.
- RNG: decode uses no RNG.
- Errors: `native_payload` bounds checks and dtype/error classes were unchanged.
- Golden output: existing native-decode fixture remains pinned at sha256
  `93332e6e21332f43535b64ab5fe3224f22213becb95cb4d3b6e0ed9888dbe943`.

## Result

Rejected. The only committed diff for this bead is this negative-result artifact
and the `.beads` closeout; no `ft-serialize` source change remains.
