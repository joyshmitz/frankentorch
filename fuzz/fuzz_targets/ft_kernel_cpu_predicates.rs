#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{
    isfinite_tensor_contiguous_f64, isinf_tensor_contiguous_f64,
    isnan_tensor_contiguous_f64,
};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let ndim = usize::from(data[0] % 5);
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

    // Mix adversarial special values with finite values so all
    // three predicates have positives and negatives every run.
    let storage: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(ndim + i) % body.len()];
            match raw % 12 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                3 => 0.0,
                4 => -0.0,
                _ => f64::from(raw as i8) / 30.0,
            }
        })
        .collect();

    let nan_out = match isnan_tensor_contiguous_f64(&storage, &meta) {
        Ok(v) => v,
        Err(_) => return,
    };
    let inf_out = match isinf_tensor_contiguous_f64(&storage, &meta) {
        Ok(v) => v,
        Err(_) => return,
    };
    let fin_out = match isfinite_tensor_contiguous_f64(&storage, &meta) {
        Ok(v) => v,
        Err(_) => return,
    };
    assert_eq!(nan_out.len(), numel, "isnan length");
    assert_eq!(inf_out.len(), numel, "isinf length");
    assert_eq!(fin_out.len(), numel, "isfinite length");

    // Per-cell invariants:
    // 1. Each output is exactly 0.0 or 1.0 (no other bit patterns).
    // 2. Mutual exclusion: exactly one of (isnan, isinf, isfinite)
    //    is 1.0; the others are 0.0. Their sum is always 1.0.
    for i in 0..numel {
        for (name, v) in [("isnan", nan_out[i]), ("isinf", inf_out[i]), ("isfinite", fin_out[i])] {
            assert!(
                v == 0.0 || v == 1.0,
                "{name}[{i}] = {v} not in {{0.0, 1.0}}"
            );
        }
        let sum = nan_out[i] + inf_out[i] + fin_out[i];
        assert!(
            (sum - 1.0).abs() < f64::EPSILON,
            "mutual exclusion broken at [{i}]: isnan={}, isinf={}, isfinite={} (sum={})",
            nan_out[i], inf_out[i], fin_out[i], sum
        );
    }
});
