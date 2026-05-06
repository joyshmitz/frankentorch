#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::index_select_tensor_contiguous_f64;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 8;
const MAX_NUM_INDICES: usize = 64;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // Preamble: ndim, dim, num_indices, index_mode (0 = in-range,
    // 1 = wrapped-negative, 2 = mixed signed/unsigned). The 1D
    // index list is independent of the source shape — that's the
    // distinguishing invariant from gather (which couples index
    // shape to input shape).
    let ndim = usize::from(data[0] % 7);
    if ndim == 0 {
        return;
    }
    let dim = usize::from(data[1] % (ndim as u8).max(1)) % ndim;
    let num_indices = usize::from(data[2]) % (MAX_NUM_INDICES + 1);
    let index_mode = data[3] % 3;
    let body = &data[4..];

    if body.len() < ndim {
        return;
    }
    let shape: Vec<usize> = body[..ndim]
        .iter()
        .map(|b| usize::from(b % (MAX_SHAPE_DIM + 1)))
        .collect();

    // index_select normalizes wrapped indices against dim_size, so
    // dim_size = 0 is degenerate (no valid index exists). Skip.
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

    let numel = meta.numel();
    if numel > 4096 || num_indices > 4096 {
        return;
    }
    // Output volume bound: outer * num_indices * inner — exercise
    // the kernel's checked_mul guard by allowing num_indices up
    // to MAX_NUM_INDICES even when the source numel is small.
    let storage: Vec<f64> = (0..numel).map(|i| i as f64).collect();

    let dim_size = shape[dim];
    #[allow(clippy::cast_precision_loss)]
    let mut indices: Vec<f64> = Vec::with_capacity(num_indices);
    for i in 0..num_indices {
        let raw = body[(ndim + i) % body.len()] as i32;
        let v = match index_mode {
            // In-range: [0, dim_size).
            0 => (raw.unsigned_abs() as usize % dim_size) as f64,
            // Wrapped-negative: [-dim_size, 0) — exercises
            // normalize_wrapped_index_value's negative branch.
            1 => -(((raw.unsigned_abs() as usize % dim_size) as f64) + 1.0),
            // Mixed: alternate sign on each index slot. Catches
            // sign-handling bugs in the per-index normalizer.
            _ => {
                if i % 2 == 0 {
                    (raw.unsigned_abs() as usize % dim_size) as f64
                } else {
                    -(((raw.unsigned_abs() as usize % dim_size) as f64) + 1.0)
                }
            }
        };
        indices.push(v);
    }

    if let Ok(output) = index_select_tensor_contiguous_f64(&storage, &meta, dim, &indices) {
        // Output length is outer * num_indices * inner. Recompute
        // independently to catch off-by-ones in the kernel's
        // pre-allocation. ndim >= 1 here, so we can split safely.
        let outer: usize = shape[..dim].iter().product();
        let inner: usize = shape[dim + 1..].iter().product();
        let expected = outer
            .checked_mul(num_indices)
            .and_then(|v| v.checked_mul(inner));
        if let Some(exp) = expected {
            assert_eq!(
                output.len(),
                exp,
                "index_select output length must equal outer * num_indices * inner"
            );
        }
    }
});
