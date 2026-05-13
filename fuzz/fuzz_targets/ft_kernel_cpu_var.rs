#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{std_dim_tensor_contiguous_f64, var_dim_tensor_contiguous_f64};
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
    let dim = usize::from(data[1] % (ndim as u8).max(1)) % ndim;
    let body = &data[2..];

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

    let storage: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(ndim + i) % body.len()] as i32;
            f64::from(raw - 128) / 40.0
        })
        .collect();

    let var_out = match var_dim_tensor_contiguous_f64(&storage, &meta, dim) {
        Ok(out) => out,
        Err(_) => return,
    };
    let std_out = match std_dim_tensor_contiguous_f64(&storage, &meta, dim) {
        Ok(out) => out,
        Err(_) => return,
    };
    assert_eq!(
        var_out.len(),
        std_out.len(),
        "var and std output lengths must match"
    );

    // Expected out_numel: 0 if any shape dim is 0, else
    // numel / dim_size (no division — multiply non-dim dims).
    let mut out_numel: usize = 1;
    let mut zero_seen = false;
    for (d, &s) in shape.iter().enumerate() {
        if s == 0 {
            zero_seen = true;
            break;
        }
        if d != dim {
            out_numel = match out_numel.checked_mul(s) {
                Some(v) => v,
                None => return,
            };
        }
    }
    if zero_seen {
        assert_eq!(var_out.len(), 0, "var of zero-dim tensor must be empty");
        return;
    }
    assert_eq!(
        var_out.len(),
        out_numel,
        "var output length must equal numel/dim_size"
    );

    let reduce_size = shape[dim];

    // Edge case: reduce_size < 2 must return NaN for every cell
    // (Bessel correction divides by reduce_size - 1).
    if reduce_size < 2 {
        for (i, &v) in var_out.iter().enumerate() {
            assert!(v.is_nan(), "var[{i}] with reduce_size={reduce_size} should be NaN, got {v}");
        }
        for (i, &v) in std_out.iter().enumerate() {
            assert!(v.is_nan(), "std[{i}] with reduce_size={reduce_size} should be NaN, got {v}");
        }
        return;
    }

    // For reduce_size >= 2 with finite input:
    //   - var >= 0 (within FP slack)
    //   - std == sqrt(var) within ULP tolerance
    const EPS_NEG: f64 = 1e-12;
    for (i, (&v_var, &v_std)) in var_out.iter().zip(std_out.iter()).enumerate() {
        if !v_var.is_finite() || !v_std.is_finite() {
            continue;
        }
        assert!(
            v_var >= -EPS_NEG,
            "var[{i}] = {v_var} should be >= 0 (dim={dim}, reduce_size={reduce_size})"
        );
        let expected_std = v_var.max(0.0).sqrt();
        let scale = expected_std.abs().max(1.0);
        assert!(
            (v_std - expected_std).abs() <= 16.0 * f64::EPSILON * scale,
            "std[{i}] = {v_std}, sqrt(var[{i}]) = {expected_std}"
        );
    }
});
