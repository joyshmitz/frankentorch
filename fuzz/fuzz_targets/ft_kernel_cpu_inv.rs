#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{inv_tensor_contiguous_f64, matmul_tensor_contiguous_f64};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let n = usize::from(2 + (data[0] % 5)); // 2..=6
    let body = &data[1..];

    // Build matrix biased toward diagonal-dominance to keep
    // condition number reasonable. Pure random matrices can be
    // near-singular and that's fine — we'll skip the A @ inv(A)
    // check when inversion fails; the kernel's error path is its
    // own valid behavior.
    let mut matrix: Vec<f64> = (0..(n * n))
        .map(|i| {
            let raw = body[i % body.len().max(1)] as i32;
            (raw - 128) as f64 / 64.0
        })
        .collect();
    // Diagonal dominance: add n*1.0 to each diagonal element.
    for i in 0..n {
        matrix[i * n + i] += n as f64;
    }

    let shape = vec![n, n];
    let meta = TensorMeta::from_shape_and_strides(
        shape.clone(),
        ft_core::contiguous_strides(&shape),
        0,
        DType::F64,
        Device::Cpu,
    )
    .ok();
    let meta = match meta {
        Some(m) => m,
        None => return,
    };

    let inv = match inv_tensor_contiguous_f64(&matrix, &meta) {
        Ok(v) => v,
        Err(_) => return, // singular or out-of-range; inversion correctly errored
    };
    assert_eq!(inv.len(), n * n, "inv output length");
    for (i, &v) in inv.iter().enumerate() {
        assert!(v.is_finite(), "inv[{i}] = {v} non-finite for diagonally-dominant input");
    }

    // Verify A @ inv(A) ≈ I within bounded ULP.
    let product = match matmul_tensor_contiguous_f64(&matrix, &inv, &meta, &meta) {
        Ok(v) => v,
        Err(_) => return,
    };
    assert_eq!(product.len(), n * n, "A @ inv(A) length");

    // Bound: for a diagonally-dominant n x n matrix, the condition
    // number is bounded. Use n^3 * EPSILON * |M|_max as a generous
    // ULP bound for the matmul step.
    let abs_max = matrix.iter().chain(inv.iter())
        .fold(0.0_f64, |acc, &v| acc.max(v.abs())).max(1.0);
    let bound = (n as f64).powi(3) * f64::EPSILON * abs_max * 64.0 + 1e-9;
    for i in 0..n {
        for j in 0..n {
            let got = product[i * n + j];
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = (got - expected).abs();
            assert!(
                diff <= bound,
                "(A @ inv(A))[{i},{j}] = {got}, expected {expected}, diff = {diff:e}, bound = {bound:e}"
            );
        }
    }
});
