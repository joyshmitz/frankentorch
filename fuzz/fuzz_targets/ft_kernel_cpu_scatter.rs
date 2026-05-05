#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::scatter_tensor_contiguous_f64;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // Preamble: ndim, dim, idx_dim_override.
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
    let mut idx_shape = shape.clone();
    idx_shape[dim] = idx_dim_override;

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

    // In-range index values cycled from body.
    let dim_size = shape[dim];
    let mut index = Vec::with_capacity(idx_numel);
    for i in 0..idx_numel {
        let raw = body[(ndim + i) % body.len()];
        let v = (usize::from(raw) % dim_size) as f64;
        index.push(v);
    }

    // src must have at least idx_numel elements; pre-fill with
    // arbitrary deterministic values so the kernel reads exercise
    // the indexed-write path.
    let src: Vec<f64> = (0..idx_numel).map(|i| i as f64).collect();

    if let Ok(output) = scatter_tensor_contiguous_f64(&storage, &meta, dim, &index, &idx_meta, &src)
    {
        // On success, output length must equal input numel —
        // recompute independently to catch allocation off-by-ones.
        assert_eq!(
            output.len(),
            numel,
            "scatter output length must equal input numel"
        );
    }
});
