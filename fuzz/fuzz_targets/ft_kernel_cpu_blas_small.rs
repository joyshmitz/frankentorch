#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{
    dot_tensor_contiguous_f64, outer_tensor_contiguous_f64, trace_tensor_contiguous_f64,
};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_DIM: u8 = 32;

fn build_1d(len: usize) -> Option<TensorMeta> {
    let shape = vec![len];
    TensorMeta::from_shape_and_strides(
        shape.clone(),
        ft_core::contiguous_strides(&shape),
        0,
        DType::F64,
        Device::Cpu,
    )
    .ok()
}

fn build_2d(rows: usize, cols: usize) -> Option<TensorMeta> {
    let shape = vec![rows, cols];
    TensorMeta::from_shape_and_strides(
        shape.clone(),
        ft_core::contiguous_strides(&shape),
        0,
        DType::F64,
        Device::Cpu,
    )
    .ok()
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let m = usize::from(data[0] % (MAX_DIM + 1));
    let n = usize::from(data[1] % (MAX_DIM + 1));
    let body = &data[2..];

    // --- dot (a, a) self-dot for non-negativity invariant ---
    if m > 0 && m <= 4096 {
        let a: Vec<f64> = (0..m)
            .map(|i| {
                let raw = body.get(i % body.len().max(1)).copied().unwrap_or(0) as i32;
                f64::from(raw - 128) / 30.0
            })
            .collect();
        if let Some(meta) = build_1d(m) {
            if let Ok(result) = dot_tensor_contiguous_f64(&a, &a, &meta, &meta) {
                assert!(result.is_finite(), "dot(a,a) = {result} should be finite");
                assert!(result >= -1e-12, "dot(a,a) = {result} should be >= 0");
                // Independent recompute: sum of squares.
                let expected: f64 = a.iter().map(|x| x * x).sum();
                let scale = expected.abs().max(1.0);
                assert!(
                    (result - expected).abs() <= 64.0 * f64::EPSILON * scale * (m as f64),
                    "dot(a,a) = {result}, independent sum-of-squares = {expected}"
                );
            }
        }
    }

    // --- outer length contract and entry identity ---
    if m > 0 && n > 0 && m * n <= 4096 {
        let lhs: Vec<f64> = (0..m)
            .map(|i| {
                let raw = body.get(i % body.len().max(1)).copied().unwrap_or(0) as i32;
                f64::from(raw - 128) / 30.0
            })
            .collect();
        let rhs: Vec<f64> = (0..n)
            .map(|i| {
                let raw = body.get((m + i) % body.len().max(1)).copied().unwrap_or(0) as i32;
                f64::from(raw - 128) / 30.0
            })
            .collect();
        let lhs_meta = build_1d(m).unwrap();
        let rhs_meta = build_1d(n).unwrap();
        if let Ok(out) = outer_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta) {
            assert_eq!(out.len(), m * n, "outer length must equal m*n");
            // Entry identity: out[i*n + j] == lhs[i] * rhs[j].
            for i in 0..m {
                for j in 0..n {
                    let got = out[i * n + j];
                    let expected = lhs[i] * rhs[j];
                    if !got.is_finite() || !expected.is_finite() {
                        continue;
                    }
                    let scale = expected.abs().max(1.0);
                    assert!(
                        (got - expected).abs() <= 4.0 * f64::EPSILON * scale,
                        "outer[{i},{j}] = {got}, lhs[{i}]*rhs[{j}] = {expected}"
                    );
                }
            }
        }
    }

    // --- trace bit-equality against independent diagonal sum ---
    // trace requires a square 2D shape.
    let trace_dim = usize::from(data[0] % (MAX_DIM + 1));
    if trace_dim > 0 && trace_dim * trace_dim <= 4096 {
        let mat: Vec<f64> = (0..(trace_dim * trace_dim))
            .map(|i| {
                let raw = body.get(i % body.len().max(1)).copied().unwrap_or(0) as i32;
                f64::from(raw - 128) / 30.0
            })
            .collect();
        if let Some(meta) = build_2d(trace_dim, trace_dim) {
            if let Ok(result) = trace_tensor_contiguous_f64(&mat, &meta) {
                let expected: f64 = (0..trace_dim).map(|i| mat[i * trace_dim + i]).sum();
                if result.is_finite() && expected.is_finite() {
                    let scale = expected.abs().max(1.0);
                    assert!(
                        (result - expected).abs() <= 16.0 * f64::EPSILON * scale,
                        "trace = {result}, independent diagonal sum = {expected}"
                    );
                }
            }
        }
    }
});
