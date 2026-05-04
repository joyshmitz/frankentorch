#![no_main]

use ft_core::{DType, Device, TensorMeta};
use libfuzzer_sys::fuzz_target;

// Cap fuzz inputs at 256 bytes — the target reads at most 8-rank shape
// + stride pairs (each 1-byte size + 1-byte stride = 16 bytes) plus
// 4 bytes of dtype/storage_offset preamble. Larger inputs would just
// be ignored by the take(8) layout below.
const MAX_TENSOR_META_INPUT_BYTES: usize = 256;

// Cap each shape dim to 32 to keep numel() tractable for the
// fuzzer (worst-case rank-8 = 32^8 ≈ 1 trillion still fits in usize
// on 64-bit). The validator's checked arithmetic should reject any
// overflow without panicking; we want the fuzzer iterating fast on
// the inputs the validator would actually walk.
const MAX_SHAPE_DIM: u8 = 32;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 || data.len() > MAX_TENSOR_META_INPUT_BYTES {
        return;
    }

    // Preamble: 1 byte dtype index, 3 bytes storage_offset (LE).
    let dtype = match data[0] % 6 {
        0 => DType::F64,
        1 => DType::F32,
        2 => DType::F16,
        3 => DType::BF16,
        4 => DType::Complex64,
        _ => DType::Complex128,
    };
    let storage_offset =
        usize::from(data[1]) | (usize::from(data[2]) << 8) | (usize::from(data[3]) << 16);

    // Decode shape/stride pairs from the rest of the input. Rank ≤ 8.
    let pairs: Vec<(u8, u8)> = data[4..]
        .chunks_exact(2)
        .take(8)
        .map(|pair| (pair[0] % (MAX_SHAPE_DIM + 1), pair[1]))
        .collect();
    let shape: Vec<usize> = pairs.iter().map(|(s, _)| usize::from(*s)).collect();
    let strides: Vec<usize> = pairs.iter().map(|(_, st)| usize::from(*st)).collect();

    // The validator's checked_mul / checked_add must catch every
    // overflow without panicking. A panic here means a missing
    // pre-condition check.
    if let Ok(meta) =
        TensorMeta::from_shape_and_strides(shape, strides, storage_offset, dtype, Device::Cpu)
    {
        // Post-validate accessors must not panic on any meta the
        // validator accepted. numel uses its own checked arithmetic
        // and saturates to usize::MAX on overflow — both is_contiguous
        // and the cheap accessors should be panic-free for any
        // validated meta.
        let _ = meta.numel();
        let _ = meta.is_contiguous();
        let _ = meta.shape();
        let _ = meta.strides();
        let _ = meta.storage_offset();
        let _ = meta.dtype();
        let _ = meta.device();
    }
});
