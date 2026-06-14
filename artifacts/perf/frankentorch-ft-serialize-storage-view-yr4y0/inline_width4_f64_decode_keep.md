# frankentorch-ft-serialize-storage-view-yr4y0

## Verdict

KEEP: native width-4 F64 decode now materializes `TensorStorage::F64Inline4([f64; 4])` instead of allocating one `Vec<f64>` per tensor.

Score: `2.54 = Impact 1.27 x Confidence 4.0 / Effort 2.0`.

## Benchmark

Target: `native_state_dict/decode_many_small_f64_1024x4`

Same worker: `vmi1293453`

- Baseline, inline branch disabled: `[355.39 us 365.38 us 377.13 us]`
- Candidate, inline branch enabled: `[278.56 us 287.41 us 295.62 us]`
- Median speedup: `365.38 / 287.41 = 1.27x`

## Isomorphism

- BTreeMap key ordering unchanged; decode still inserts by key into the same map.
- Duplicate key and malformed native payload behavior unchanged outside the width-4 F64 storage representation.
- Width-4 F64 raw bits are still decoded with `from_le_bytes` in the same element order.
- Tensor dtype, shape, device, storage offset validation, and contiguous accessors are preserved by `DenseTensor::from_typed_storage`.
- `storage()`, `typed_storage().as_f64()`, `contiguous_values()`, and `contiguous_values_as_f64()` expose the same four values for inline tensors.
- Mutation remains tensor-local: cloned inline tensors copy the four values before `update_contiguous_values`.
- No RNG, floating-point arithmetic, tie-breaking, or ordering policy changed.

## Proof

- `cargo test -j 1 -p ft-core f64_inline -- --nocapture`: passed `2/2` on `vmi1293453`.
- `cargo test -j 1 -p ft-serialize native_ -- --nocapture`: passed `16/16` on `vmi1293453`.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`: passed for all present fixtures.
- `cargo check -j 1 -p ft-core -p ft-serialize --all-targets`: passed on `vmi1293453`.
- `cargo clippy -j 1 -p ft-core -p ft-serialize --all-targets -- -D warnings`: passed on `vmi1293453`.
- `cargo fmt -p ft-core -p ft-serialize --check`: passed locally.
