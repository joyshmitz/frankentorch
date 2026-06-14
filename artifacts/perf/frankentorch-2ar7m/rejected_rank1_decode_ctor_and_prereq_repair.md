# frankentorch-2ar7m closeout: rejected rank-1 f64 decode constructor

## Target

- Bead: `frankentorch-2ar7m`
- Benchmark: `native_state_dict/decode_many_small_f64_1024x4`
- Crate: `ft-serialize`
- Profile-backed source: `artifacts/perf/frankentorch-ftserialize-f64decode/rejected_borrowed_utf8_key.md`

## Prerequisite repair kept

The initial RCH baseline could not compile `ft-serialize` against the current
`franken-kernel` API:

- `InactivationDecoder::repair_equation` now returns `Result`.
- `DecodeProof::content_hash()` now returns a 32-byte `ProofHash`, while
  FrankenTorch's persisted `DecodeProofArtifact` still exposes `proof_hash: u64`
  and `proof_hash_hex: det64:<16 hex>`.

Kept source change:

- Convert `repair_equation` failures into `SerializeError::RaptorQFailure`.
- Derive the existing deterministic `u64` proof hash from the first eight bytes
  of the new proof hash and preserve the existing `det64:` string contract.

## Rejected performance lever

One lever tried after the repaired baseline:

- Add a safe `DenseTensor::from_contiguous_rank1_values(Vec<f64>, Device)`
  constructor in `ft-core`.
- Route parser-validated rank-1 f64 native state-dict tensors through it.

The hunk preserved behavior in the focused golden decode test, but it did not
clear the performance gate and was removed.

## Benchmark evidence

Worker: `vmi1149989`.

Baseline from repaired build state:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 5 --sample-size 20
RCH selected worker: vmi1149989
native_state_dict/decode_many_small_f64_1024x4
time: [264.40 us 277.28 us 293.40 us]
```

Candidate:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 5 --sample-size 20
RCH selected worker: vmi1149989
native_state_dict/decode_many_small_f64_1024x4
time: [275.36 us 290.10 us 304.62 us]
```

Result: `277.28 us -> 290.10 us`, median ratio `0.956x`; rejected.

Score: `0.0 = Impact 0.0 x Confidence 0.85 / Effort 0.5`.

## Behavior and gates

- Repaired crate check: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -j 1 -p ft-serialize --all-targets` passed on `vmi1227854`.
- Candidate golden decode proof before rejection: `native_decode_many_small_f64_golden_summary_matches_fixture` passed on `vmi1227854`.
- Final-source proof-hash test: `decode_proof_hash_is_deterministic` passed on `vmi1149989`.
- Final-source clippy: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -j 1 -p ft-serialize --all-targets -- -D warnings` passed on `vmi1227854`.
- Final-source fmt: `cargo fmt -p ft-serialize --check` passed locally.

Isomorphism:

- Ordering: final source makes no state-dict decode-path change.
- Duplicate-key behavior: final source makes no state-dict decode-path change.
- Floating point/raw bits: final source makes no state-dict decode-path change.
- RNG: not involved.
- RaptorQ proof hash: deterministic u64 contract preserved by the focused proof-hash test.

## Next deeper primitive

Do not retry constructor-level rank-1 decode micro-levers. The next attack should
be a true structural native decode primitive:

- build a compact borrowed record index for native state-dict entries;
- sort/check the record index without constructing `DenseTensor` values;
- bulk materialize final owned keys and tensors only after duplicate/order and
  malformed-input checks are complete;
- preserve exact duplicate-key failure point, invalid UTF-8 diagnostics,
  truncated payload diagnostics, raw f64 bits, and final `BTreeMap` ordering.

Target ratio: at least `1.25x` on `native_state_dict/decode_many_small_f64_1024x4`.
