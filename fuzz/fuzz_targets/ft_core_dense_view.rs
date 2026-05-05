#![no_main]

use ft_core::{DType, DenseTensor, Device, TensorMeta};
use libfuzzer_sys::fuzz_target;

// Cap fuzz inputs at 256 bytes — the target reads at most 8 source-shape
// dim bytes + 8 view-shape dim bytes, plus a small preamble. Larger
// inputs would just be ignored by the take(8) layouts below.
const MAX_INPUT_BYTES: usize = 256;

// Cap each shape dim at 16 to keep numel tractable for the fuzzer
// (worst-case rank-8 source = 16^8 ≈ 4 billion, still fits in usize on
// 64-bit). The validator's checked arithmetic should reject any
// overflow without panicking; we want the fuzzer iterating fast on
// inputs that exercise the validator without OOMing the worker.
const MAX_SHAPE_DIM: u8 = 16;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // Preamble: 1 byte source rank, 1 byte view rank. Upper-bounded
    // at 8 to keep the input space tractable.
    let src_rank = usize::from(data[0] % 9);
    let view_rank = usize::from(data[1] % 9);
    let body = &data[2..];

    if body.len() < src_rank + view_rank {
        return;
    }
    let src_dims: Vec<usize> = body[..src_rank]
        .iter()
        .map(|b| usize::from(b % (MAX_SHAPE_DIM + 1)))
        .collect();
    let view_dims: Vec<usize> = body[src_rank..src_rank + view_rank]
        .iter()
        .map(|b| usize::from(b % (MAX_SHAPE_DIM + 1)))
        .collect();

    // Build the source DenseTensor with the chosen shape, F64
    // contiguous storage. Any failure to construct (e.g. invalid
    // rank-0 with non-empty storage) is fine — we only care about
    // the view() validator below.
    let meta = match TensorMeta::from_shape_and_strides(
        src_dims.clone(),
        ft_core::contiguous_strides(&src_dims),
        0,
        DType::F64,
        Device::Cpu,
    ) {
        Ok(meta) => meta,
        Err(_) => return,
    };
    let numel = meta.numel();
    // Cap storage allocation at 4096 elements (32 KiB) so the fuzzer
    // doesn't burn iteration budget on huge allocations the validator
    // would accept.
    if numel > 4096 {
        return;
    }
    let storage = vec![0.0_f64; numel];
    let tensor = match DenseTensor::from_storage(meta, storage) {
        Ok(t) => t,
        Err(_) => return,
    };

    // The actual fuzz target: DenseTensor::view validates new_shape
    // via checked_shape_numel + numel parity. Any panic here means
    // a missing pre-condition check.
    if let Ok(viewed) = tensor.view(view_dims) {
        // Post-validate accessors must not panic on any shape the
        // validator accepted.
        let _ = viewed.meta().numel();
        let _ = viewed.meta().is_contiguous();
        let _ = viewed.meta().shape();
        let _ = viewed.meta().strides();
        let _ = viewed.meta().dtype();
        let _ = viewed.shares_storage_with(&tensor);
    }
});
