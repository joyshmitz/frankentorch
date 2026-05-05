#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::gather_tensor_contiguous_f64;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
// Cap each shape dim at 8 to keep numel tractable for the fuzzer.
const MAX_SHAPE_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // Preamble: ndim (shared between input and index per the
    // gather contract), dim (index into shape), and index-dim-size
    // override (lets fuzzer explore both same-size and different-
    // size dim cases).
    let ndim = usize::from(data[0] % 7);
    if ndim == 0 {
        return;
    }
    let dim = usize::from(data[1] % (ndim as u8).max(1)) % ndim;
    let idx_dim_override = usize::from(data[2] % (MAX_SHAPE_DIM + 1));
    let body = &data[3..];

    if body.len() < ndim + 1 {
        return;
    }
    let shape: Vec<usize> = body[..ndim]
        .iter()
        .map(|b| usize::from(b % (MAX_SHAPE_DIM + 1)))
        .collect();
    // Index shape: same as input shape except the gather dim is
    // replaced with idx_dim_override.
    let mut idx_shape = shape.clone();
    idx_shape[dim] = idx_dim_override;

    // Skip degenerate cases the kernel rejects ahead of time.
    if shape[dim] == 0 {
        return;
    }

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
    let idx_meta = match TensorMeta::from_shape_and_strides(
        idx_shape.clone(),
        ft_core::contiguous_strides(&idx_shape),
        0,
        DType::F64,
        Device::Cpu,
    ) {
        Ok(meta) => meta,
        Err(_) => return,
    };
    let numel = meta.numel();
    let idx_numel = idx_meta.numel();
    if numel > 4096 || idx_numel > 4096 {
        return;
    }
    let storage = vec![0.0_f64; numel];

    // Index values in [0, dim_size). Cycle through the body bytes.
    let dim_size = shape[dim];
    let mut index = Vec::with_capacity(idx_numel);
    for i in 0..idx_numel {
        let raw = body[(ndim + i) % body.len()];
        let v = (usize::from(raw) % dim_size) as f64;
        index.push(v);
    }

    if let Ok(output) = gather_tensor_contiguous_f64(&storage, &meta, dim, &index, &idx_meta) {
        // On success, output length must equal index numel —
        // recompute independently to catch allocation off-by-ones
        // in the kernel.
        assert_eq!(
            output.len(),
            idx_numel,
            "gather output length must equal index numel"
        );
    }
});
