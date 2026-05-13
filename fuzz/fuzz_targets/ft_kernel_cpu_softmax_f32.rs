#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{
    log_softmax_dim_tensor_contiguous_f32, softmax_dim_tensor_contiguous_f32,
};
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
        DType::F32,
        Device::Cpu,
    ) {
        Ok(meta) => meta,
        Err(_) => return,
    };
    let numel = meta.numel();
    if numel > 4096 {
        return;
    }

    let storage: Vec<f32> = (0..numel)
        .map(|i| {
            let raw = body[(ndim + i) % body.len()] as i32;
            (raw - 128) as f32 / 25.0
        })
        .collect();

    if numel == 0 {
        let s = softmax_dim_tensor_contiguous_f32(&storage, &meta, dim);
        let ls = log_softmax_dim_tensor_contiguous_f32(&storage, &meta, dim);
        assert!(s.is_ok() && ls.is_ok(), "empty softmax_f32 must succeed");
        return;
    }

    let softmax = match softmax_dim_tensor_contiguous_f32(&storage, &meta, dim) {
        Ok(out) => out,
        Err(_) => return,
    };
    assert_eq!(softmax.len(), numel, "softmax_f32 length mismatch");

    let log_softmax = match log_softmax_dim_tensor_contiguous_f32(&storage, &meta, dim) {
        Ok(out) => out,
        Err(_) => return,
    };
    assert_eq!(log_softmax.len(), numel, "log_softmax_f32 length mismatch");

    // f32-widened tolerances.
    const ELEM_EPS: f32 = 1e-5;
    for (idx, &v) in softmax.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        assert!(
            v >= -ELEM_EPS && v <= 1.0 + ELEM_EPS,
            "softmax_f32[{idx}] = {v} outside [0, 1]"
        );
    }
    for (idx, &v) in log_softmax.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        assert!(v <= ELEM_EPS, "log_softmax_f32[{idx}] = {v} should be <= 0");
    }

    let dim_size = shape[dim];
    if dim_size == 0 {
        return;
    }
    let inner_size: usize = shape[dim + 1..].iter().product();
    let outer_size: usize = shape[..dim].iter().product();
    // f32 sum-to-1 has much wider tolerance — N ULPs of f32 EPSILON
    // (f32_eps ≈ 1.19e-7).
    const SUM_EPS: f32 = 1e-3;
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut sum = 0.0_f32;
            let mut any_nonfinite = false;
            for d in 0..dim_size {
                let flat = outer * dim_size * inner_size + d * inner_size + inner;
                let v = softmax[flat];
                if !v.is_finite() {
                    any_nonfinite = true;
                    break;
                }
                sum += v;
            }
            if any_nonfinite {
                continue;
            }
            assert!(
                (sum - 1.0).abs() < SUM_EPS,
                "softmax_f32 sum-along-dim={dim} at outer={outer} inner={inner} = {sum}"
            );
        }
    }
});
