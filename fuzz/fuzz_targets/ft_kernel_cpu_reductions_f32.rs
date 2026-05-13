#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{
    max_dim_tensor_contiguous_f32, mean_dim_tensor_contiguous_f32,
    min_dim_tensor_contiguous_f32, prod_dim_tensor_contiguous_f32,
    sum_dim_tensor_contiguous_f32,
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
            (raw - 128) as f32 / 96.0
        })
        .collect();

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
        return;
    }

    let sum_out = match sum_dim_tensor_contiguous_f32(&storage, &meta, dim) {
        Ok(v) => v,
        Err(_) => return,
    };
    let mean_out = match mean_dim_tensor_contiguous_f32(&storage, &meta, dim) {
        Ok(v) => v,
        Err(_) => return,
    };
    let prod_out = match prod_dim_tensor_contiguous_f32(&storage, &meta, dim) {
        Ok(v) => v,
        Err(_) => return,
    };
    let (max_vals, _max_idx) = match max_dim_tensor_contiguous_f32(&storage, &meta, dim) {
        Ok(v) => v,
        Err(_) => return,
    };
    let (min_vals, _min_idx) = match min_dim_tensor_contiguous_f32(&storage, &meta, dim) {
        Ok(v) => v,
        Err(_) => return,
    };

    assert_eq!(sum_out.len(), out_numel, "sum_f32 length");
    assert_eq!(mean_out.len(), out_numel, "mean_f32 length");
    assert_eq!(prod_out.len(), out_numel, "prod_f32 length");
    assert_eq!(max_vals.len(), out_numel, "max_f32 length");
    assert_eq!(min_vals.len(), out_numel, "min_f32 length");

    // f32 has wider FP tolerance — use 1e-4 for the max>=mean>=min
    // chain (catches gross errors while permitting f32 rounding noise).
    const EPS: f32 = 1e-4;
    for i in 0..out_numel {
        if !max_vals[i].is_finite() || !mean_out[i].is_finite() || !min_vals[i].is_finite() {
            continue;
        }
        assert!(
            max_vals[i] >= mean_out[i] - EPS,
            "max_f32[{i}]={} < mean_f32[{i}]={}", max_vals[i], mean_out[i]
        );
        assert!(
            mean_out[i] >= min_vals[i] - EPS,
            "mean_f32[{i}]={} < min_f32[{i}]={}", mean_out[i], min_vals[i]
        );
    }
});
