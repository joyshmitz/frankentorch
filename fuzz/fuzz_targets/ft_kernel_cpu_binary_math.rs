#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{
    clamp_tensor_contiguous_f64, fmod_tensor_contiguous_f64,
    pow_tensor_contiguous_f64, remainder_tensor_contiguous_f64,
};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let ndim = usize::from(data[0] % 5);
    if ndim == 0 {
        return;
    }
    // Scalar exponent for pow.
    let exponent = match data[1] % 8 {
        0 => 0.0,
        1 => 1.0,
        2 => 2.0,
        3 => -1.0,
        4 => 0.5,
        b => f64::from(b as i32 - 4) / 2.0,
    };
    // Scalar bounds for clamp; ensure min <= max.
    let raw_min = f64::from(data[2] as i32 - 128) / 50.0;
    let raw_max = f64::from(data[3] as i32 - 128) / 50.0;
    let (min_val, max_val) = if raw_min <= raw_max {
        (raw_min, raw_max)
    } else {
        (raw_max, raw_min)
    };
    let body = &data[4..];

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

    let lhs: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(ndim + i) % body.len()] as i32;
            f64::from(raw - 128) / 40.0
        })
        .collect();
    // rhs uses a different offset, and biased toward non-zero so
    // fmod/remainder don't trivially produce NaN.
    let rhs: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(ndim + 2 * i + 1) % body.len()] as i32;
            let v = f64::from(raw - 128) / 40.0;
            if v.abs() < 1e-6 { 1.0 } else { v }
        })
        .collect();

    // --- pow ---
    if let Ok(out) = pow_tensor_contiguous_f64(&lhs, &meta, exponent) {
        assert_eq!(out.len(), numel, "pow output length");
        if exponent == 0.0 {
            for (i, (&o, &x)) in out.iter().zip(lhs.iter()).enumerate() {
                if !x.is_finite() || x == 0.0 {
                    continue; // 0^0 and inf^0 are special; skip
                }
                // pow(x, 0) == 1 for any finite non-zero x.
                assert!(
                    (o - 1.0).abs() < 1e-12,
                    "pow({x}, 0)[{i}] = {o}, expected 1.0"
                );
            }
        }
    }

    // --- clamp ---
    if let Ok(out) = clamp_tensor_contiguous_f64(&lhs, &meta, min_val, max_val) {
        assert_eq!(out.len(), numel, "clamp output length");
        for (i, (&o, &x)) in out.iter().zip(lhs.iter()).enumerate() {
            if x.is_nan() {
                assert!(o.is_nan(), "clamp(NaN)[{i}] = {o}, expected NaN");
                continue;
            }
            // Output must be in [min_val, max_val] when min <= max.
            assert!(
                o >= min_val - 1e-12 && o <= max_val + 1e-12,
                "clamp(x={x}, min={min_val}, max={max_val})[{i}] = {o}"
            );
        }
    }

    // clamp(x, x_min, x_max) where x_min == x_max == x must equal x
    // bit-exactly for finite x — degenerate-range identity.
    let exact_clamp = clamp_tensor_contiguous_f64(&lhs, &meta, 0.0, 0.0).ok();
    if let Some(out) = exact_clamp {
        for &o in &out {
            if !o.is_nan() {
                assert!(o == 0.0, "clamp to [0, 0] should yield 0, got {o}");
            }
        }
    }

    // --- fmod (C-style: result has sign of lhs) ---
    if let Ok(out) = fmod_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta) {
        assert_eq!(out.len(), numel, "fmod output length");
        for (i, ((&o, &a), &b)) in out.iter().zip(lhs.iter()).zip(rhs.iter()).enumerate() {
            if !a.is_finite() || !b.is_finite() || b == 0.0 || a == 0.0 || o == 0.0 {
                continue;
            }
            // For finite non-zero a, b: sign(fmod(a, b)) == sign(a).
            // (Skip a == 0 case: result is 0, sign is irrelevant.)
            assert!(
                a.signum() == o.signum(),
                "fmod[{i}](a={a}, b={b}) = {o}: sign should match a"
            );
        }
    }

    // --- remainder (Python-style: result has sign of rhs) ---
    if let Ok(out) = remainder_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta) {
        assert_eq!(out.len(), numel, "remainder output length");
        for (i, ((&o, &a), &b)) in out.iter().zip(lhs.iter()).zip(rhs.iter()).enumerate() {
            if !a.is_finite() || !b.is_finite() || b == 0.0 || o == 0.0 {
                continue;
            }
            assert!(
                b.signum() == o.signum(),
                "remainder[{i}](a={a}, b={b}) = {o}: sign should match b"
            );
        }
    }
});
