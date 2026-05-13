#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::matmul_tensor_contiguous_f64;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_DIM: u8 = 16;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // Preamble: m, k, n in [0, 16].
    let m = usize::from(data[0] % (MAX_DIM + 1));
    let k = usize::from(data[1] % (MAX_DIM + 1));
    let n = usize::from(data[2] % (MAX_DIM + 1));
    let body = &data[3..];

    let lhs_shape = vec![m, k];
    let rhs_shape = vec![k, n];
    let lhs_meta = match TensorMeta::from_shape_and_strides(
        lhs_shape.clone(),
        ft_core::contiguous_strides(&lhs_shape),
        0,
        DType::F64,
        Device::Cpu,
    ) {
        Ok(meta) => meta,
        Err(_) => return,
    };
    let rhs_meta = match TensorMeta::from_shape_and_strides(
        rhs_shape.clone(),
        ft_core::contiguous_strides(&rhs_shape),
        0,
        DType::F64,
        Device::Cpu,
    ) {
        Ok(meta) => meta,
        Err(_) => return,
    };

    let lhs_numel = m * k;
    let rhs_numel = k * n;
    let out_numel = m * n;
    if lhs_numel > 4096 || rhs_numel > 4096 || out_numel > 4096 {
        return;
    }

    let lhs: Vec<f64> = (0..lhs_numel)
        .map(|i| {
            let raw = body.get(i % body.len().max(1)).copied().unwrap_or(0) as i32;
            f64::from(raw - 128) / 40.0
        })
        .collect();
    let rhs: Vec<f64> = (0..rhs_numel)
        .map(|i| {
            let raw = body.get((lhs_numel + i) % body.len().max(1)).copied().unwrap_or(0) as i32;
            f64::from(raw - 128) / 40.0
        })
        .collect();

    let output = match matmul_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta) {
        Ok(out) => out,
        Err(_) => return,
    };
    assert_eq!(output.len(), out_numel, "matmul output length mismatch");

    // Finiteness for bounded input: with values in [-3.2, 3.2] and
    // k <= 16, the worst-case |sum| is bounded well below f64 max,
    // so every output cell must be finite. Catches NaN injection
    // and accidental division.
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "matmul[{i}] = {v} should be finite");
    }
});
