# INVALIDATED: frankentorch-ykpvk late reprofile note

This file was created after the shared checkout advanced to commit `933b80df`
(`perf(ft-serialize): reject ykpvk inline constructor`). The constructor source
hunk was already removed, so the late `vmi1227854` timing is a post-rejection
current-HEAD reprofile, not a keep proof for the ykpvk candidate.

Canonical ykpvk closeout remains:

`artifacts/perf/frankentorch-ykpvk/rejected_rank1_inline_constructor.md`

The stale notes below are retained only to avoid hiding the audit trail for this
invalidated local artifact.

# Stale frankentorch-ykpvk closeout draft

Bead: `frankentorch-ykpvk`

Lever: checked-by-construction rank-1 width-4 F64 CPU inline constructor, used only by the guarded native state-dict fast path.

## Profile-backed target

Primary target row:

`native_state_dict/decode_many_small_f64_1024x4`

Routing baseline from the bead description, post-f8uji on `vmi1293453`:

`[254.89 us 260.78 us 270.87 us]`

Decisive same-worker proof used the captured post-f8uji `vmi1227854` baseline from `artifacts/perf/frankentorch-f8uji/rebench_decode_many_small_f64_1024x4_current_logged.log`:

`[264.70 us 274.43 us 289.29 us]`

Candidate on `vmi1227854`:

`[213.06 us 223.00 us 232.22 us]`

Median speedup:

`274.43 / 223.00 = 1.231x` (`18.7%` faster)

Score:

`2.46 = Impact 1.231 * Confidence 4.0 / Effort 2.0`

Verdict: KEEP.

## Behavior proof

- Ordering: final output remains a `BTreeMap`; the fast path still inserts the same owned keys.
- Duplicate keys: duplicate insertion still declines the fast path and falls back to the generic parser, preserving the existing duplicate-key error.
- Malformed inputs: trailing bytes, impossible tensor counts, and guarded native truncation behavior passed the focused tests.
- Floating point: the fast path still reconstructs exact little-endian F64 payload bits before constructing the inline tensor.
- DType/shape/meta: the new constructor is hard-coded to rank-1 shape `[4]`, stride `[1]`, `numel=4`, offset `0`, `DType::F64`, `Device::Cpu`, no quantization, inline F64 storage, and version `0`.
- Tensor identity: constructor preserves the existing `NEXT_TENSOR_ID` and `NEXT_STORAGE_ID` allocation semantics.
- Tie-breaking and RNG: no comparator, sorting policy, random source, or floating-point arithmetic changed.
- Golden SHA: `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed for all present fixtures, including `ft_serialize_decode_pass19.txt`.

## Gates

- RCH `vmi1153651`: `cargo test -j 1 -p ft-core dense_tensor_rank1_f64_inline4_cpu_constructor_sets_canonical_meta -- --nocapture` passed `1/1`.
- RCH `vmi1149989`: `cargo test -j 1 -p ft-serialize native_ -- --nocapture` passed `18/18`.
- RCH `vmi1227854`: `cargo test -j 1 -p ft-serialize load_state_dict_rejects -- --nocapture` passed `3/3`.
- RCH `vmi1227854`: `cargo check -j 1 -p ft-core -p ft-serialize --all-targets` passed.
- RCH `vmi1167313`: `cargo clippy -j 1 -p ft-core -p ft-serialize --all-targets -- -D warnings` passed.
- Local: `cargo fmt -p ft-core -p ft-serialize --check` passed.
- UBS: `ubs crates/ft-core/src/lib.rs crates/ft-serialize/src/lib.rs` exited 0 with no critical findings; remaining warnings are broad existing inventories.

## Notes

This is one lever: the generic `TensorMeta::from_shape(...)` plus `DenseTensor::from_storage_f64_inline4(...)` path is replaced only for the already-canonical rank-1 width-4 F64 native fast path. All other state-dict records continue through the existing parser and validation flow.
