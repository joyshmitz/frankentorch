#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::narrow_tensor_contiguous_f64;
use libfuzzer_sys::fuzz_target;

// Cap fuzz inputs at 256 bytes — the target reads at most 8 shape
// dim bytes plus a 4-byte preamble (rank + dim + start + length).
const MAX_INPUT_BYTES: usize = 256;

// Cap each shape dim at 16 to keep numel tractable for the fuzzer
// (worst-case rank-8 = 16^8 ≈ 4 billion, still fits in usize on
// 64-bit). The kernel's checked arithmetic should reject any
// overflow without panicking.
const MAX_SHAPE_DIM: u8 = 16;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // Preamble: rank, dim, start, length (each clamped to small range).
    let rank = usize::from(data[0] % 9);
    let dim_raw = usize::from(data[1]);
    let start = usize::from(data[2] % (MAX_SHAPE_DIM + 1));
    let length = usize::from(data[3] % (MAX_SHAPE_DIM + 1));
    let body = &data[4..];

    if body.len() < rank {
        return;
    }
    let shape: Vec<usize> = body[..rank]
        .iter()
        .map(|b| usize::from(b % (MAX_SHAPE_DIM + 1)))
        .collect();

    // Build a contiguous F64 meta. Skip if any dim is zero AND
    // we're trying to narrow it (handled by the kernel's bounds
    // check, but skip ahead-of-time to keep the fuzzer iterating
    // on inputs the validator actually walks).
    let meta = match TensorMeta::from_shape_and_strides(
        shape.clone(),
        ft_core::contiguous_strides(&shape),
        0,
        DType::F64,
        Device::Cpu,
    ) {
        Ok(meta) => meta,
        Err(_) => return,
    };

    let numel = meta.numel();
    if numel > 4096 {
        return;
    }
    let storage = vec![0.0_f64; numel];

    // The actual fuzz target: narrow_tensor_contiguous_f64 validates
    // dim < ndim, start + length checked_add, and out_numel
    // checked_mul. Any panic here means a missing pre-condition.
    if let Ok(output) = narrow_tensor_contiguous_f64(&storage, &meta, dim_raw, start, length) {
        // On success, output length must equal outer * length * inner
        // where outer = product(shape[..dim]) and inner =
        // product(shape[dim+1..]). Recompute directly to catch off-
        // by-ones in the kernel's allocation logic.
        if dim_raw < shape.len() {
            let outer: usize = shape[..dim_raw].iter().product();
            let inner: usize = shape[dim_raw + 1..].iter().product();
            let expected = outer
                .checked_mul(length)
                .and_then(|p| p.checked_mul(inner))
                .unwrap_or(usize::MAX);
            assert_eq!(
                output.len(),
                expected,
                "narrow output length must equal outer * length * inner"
            );
        }
    }
});
