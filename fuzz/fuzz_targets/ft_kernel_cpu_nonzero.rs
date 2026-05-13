#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::nonzero_tensor_contiguous_f64;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let ndim = usize::from(data[0] % 7);
    if ndim == 0 {
        return;
    }
    let body = &data[1..];

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

    // Mixed input: zeros (selected out), negative zeros (also zero
    // by IEEE so NOT selected), NaN (non-zero, selected), inf,
    // finite values.
    let input: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(ndim + i) % body.len()];
            match raw % 8 {
                0 => 0.0,
                1 => -0.0,
                2 => f64::NAN,
                3 => f64::INFINITY,
                4 => f64::NEG_INFINITY,
                _ => f64::from(raw as i8) / 9.0,
            }
        })
        .collect();

    let (indices_flat, num_nonzero) = match nonzero_tensor_contiguous_f64(&input, &meta) {
        Ok(pair) => pair,
        Err(_) => return,
    };

    // Independent count: -0.0 == 0.0 under == so not counted.
    let expected_count = input.iter().filter(|x| **x != 0.0).count();
    assert_eq!(num_nonzero, expected_count, "nonzero count mismatch");

    // Flat length contract: num_nonzero * ndim.
    let expected_flat_len = num_nonzero.saturating_mul(ndim);
    assert_eq!(
        indices_flat.len(),
        expected_flat_len,
        "nonzero flat length mismatch"
    );

    // Each (ndim-stride) group of indices must decode to a
    // position where input is non-zero. Indices are stored as
    // f64 — verify they're exact non-negative integers and the
    // corresponding cell != 0.0.
    if ndim == 0 || expected_count == 0 {
        return;
    }
    // Compute strides once outside the chunk loop.
    let mut strides_back = vec![1usize; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        strides_back[d] = strides_back[d + 1].saturating_mul(shape[d + 1]);
    }
    for group in indices_flat.chunks_exact(ndim) {
        let mut flat = 0usize;
        for (d, &idx_f) in group.iter().enumerate() {
            let idx = idx_f as usize;
            assert!(
                (idx_f - idx as f64).abs() < f64::EPSILON,
                "nonzero index not integer: {idx_f}"
            );
            assert!(
                idx < shape[d],
                "nonzero index {idx} out of range for dim {d} (size {})", shape[d]
            );
            flat += idx * strides_back[d];
        }
        assert!(
            input[flat] != 0.0,
            "nonzero selected zero cell at flat {flat}: input = {}", input[flat]
        );
    }
});
