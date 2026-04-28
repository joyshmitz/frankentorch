#![no_main]

use libfuzzer_sys::fuzz_target;

// Cap fuzz inputs at 256 KiB so a pathological huge header (the safetensors
// format prefixes the JSON header with an 8-byte length) cannot stall the
// fuzzer with a single multi-megabyte allocation. The native FTSV target
// uses a similar cap; safetensors headers can be richer (per-tensor offsets,
// metadata map) so this cap is intentionally larger.
const MAX_SAFETENSORS_INPUT_BYTES: usize = 256 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_SAFETENSORS_INPUT_BYTES {
        return;
    }

    // The safetensors loader exercises:
    //   * 8-byte little-endian header length read
    //   * JSON header parse (tensor table, dtype/shape/offsets, optional metadata)
    //   * Per-tensor byte-width validation against shape numel
    //   * Per-tensor LE byte-pattern decode for F64 / F32 / F16 / BF16
    //   * Rejection of unsupported dtypes
    // load_safetensors_from_bytes returns a Result, so any panic surfaced by
    // the underlying safetensors crate (the only third-party parser FrankenTorch
    // exposes) is a crash worth catching.
    let _ = ft_serialize::load_safetensors_from_bytes(data);
});
