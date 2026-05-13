#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::topk_tensor_contiguous_f64;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 5 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // Preamble: ndim, dim, k_raw, largest_bit, sorted_bit, then shape.
    let ndim = usize::from(data[0] % 7);
    if ndim == 0 {
        return;
    }
    let dim = usize::from(data[1] % (ndim as u8).max(1)) % ndim;
    let k_raw = data[2];
    let largest = (data[3] & 1) == 1;
    let sorted = (data[4] & 1) == 1;
    let body = &data[5..];

    if body.len() < ndim {
        return;
    }
    let shape: Vec<usize> = body[..ndim]
        .iter()
        .map(|b| usize::from(b % (MAX_SHAPE_DIM + 1)))
        .collect();

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
    let dim_size = shape[dim];
    if dim_size == 0 {
        return;
    }

    // k in [0, dim_size]. k = 0 is degenerate but valid; k > dim_size
    // is rejected by the kernel (we explicitly cap to avoid wasted
    // Err paths).
    let k = usize::from(k_raw) % (dim_size + 1);

    let storage: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(ndim + i) % body.len()];
            match raw % 16 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                _ => f64::from(raw as i8) / 11.0,
            }
        })
        .collect();

    let (values, indices) =
        match topk_tensor_contiguous_f64(&storage, &meta, k, dim, largest, sorted) {
            Ok(pair) => pair,
            Err(_) => return,
        };

    // Output volume contract: outer * k * inner.
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();
    let expected = outer_size.saturating_mul(k).saturating_mul(inner_size);
    assert_eq!(values.len(), expected, "topk values len mismatch");
    assert_eq!(indices.len(), expected, "topk indices len mismatch");

    // Per-lane validation: each index in [0, dim_size); each lane's
    // k indices are distinct; values[i] bit-equals input at indices[i].
    if k == 0 {
        return;
    }
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut seen = vec![false; dim_size];
            for slot in 0..k {
                let out_idx = outer * k * inner_size + slot * inner_size + inner;
                let orig_d = indices[out_idx];
                assert!(
                    orig_d < dim_size,
                    "topk index out of range: orig_d={orig_d} dim_size={dim_size}"
                );
                assert!(
                    !seen[orig_d],
                    "topk index appears twice in lane: orig_d={orig_d}"
                );
                seen[orig_d] = true;
                let src_idx = outer * dim_size * inner_size + orig_d * inner_size + inner;
                assert_eq!(
                    storage[src_idx].to_bits(),
                    values[out_idx].to_bits(),
                    "topk value/index mismatch: input[{src_idx}] != values[{out_idx}]"
                );
            }
        }
    }
});
