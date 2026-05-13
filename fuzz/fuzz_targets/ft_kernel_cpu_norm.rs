#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::norm_dim_tensor_contiguous_f64;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let ndim = usize::from(data[0] % 7);
    if ndim == 0 {
        return;
    }
    let dim = usize::from(data[1] % (ndim as u8).max(1)) % ndim;
    // p selector: cover each special branch plus fractional.
    let p = match data[2] % 6 {
        0 => 0.0,
        1 => 1.0,
        2 => 2.0,
        3 => f64::INFINITY,
        4 => f64::NEG_INFINITY,
        _ => 0.5 + f64::from((data[2] % 16) as i32) / 8.0, // ~[0.5, 2.4]
    };
    let body = &data[3..];

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

    // Bounded finite input keeps the output non-NaN for all p
    // branches (avoids 0^0 = NaN under p=0).
    let storage: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(ndim + i) % body.len()] as i32;
            f64::from(raw - 128) / 40.0
        })
        .collect();

    let output = match norm_dim_tensor_contiguous_f64(&storage, &meta, p, dim) {
        Ok(out) => out,
        Err(_) => return,
    };

    // Expected output length: numel / dim_size (or 0 when shape
    // has any zero extent). Compute independently to catch
    // allocation off-by-ones.
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
        assert_eq!(
            output.len(),
            0,
            "norm output for zero-dim tensor must be empty"
        );
        return;
    }
    assert_eq!(
        output.len(),
        out_numel,
        "norm output length must equal numel/dim_size"
    );

    // Range invariant: norm of finite input must be finite and
    // non-negative (with small FP slack).
    const EPS: f64 = 1e-12;
    for (i, &v) in output.iter().enumerate() {
        assert!(
            v.is_finite(),
            "norm[{i}] = {v} should be finite (p={p}, dim={dim})"
        );
        assert!(
            v >= -EPS,
            "norm[{i}] = {v} should be non-negative (p={p}, dim={dim})"
        );
    }
});
