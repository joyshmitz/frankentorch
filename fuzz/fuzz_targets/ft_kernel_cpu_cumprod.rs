#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{cumprod_backward_tensor_contiguous_f64, cumprod_tensor_contiguous_f64};
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

    // Input range scaled into [-1, 1] so the running product
    // doesn't immediately saturate to 0 or inf. Include
    // occasional zeros to exercise the zero-propagation path.
    let storage: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(ndim + i) % body.len()];
            match raw % 16 {
                0 => 0.0,
                _ => (f64::from(raw as i8) / 128.0).clamp(-1.0, 1.0),
            }
        })
        .collect();

    let forward_output = match cumprod_tensor_contiguous_f64(&storage, &meta, dim) {
        Ok(out) => out,
        Err(_) => return,
    };
    assert_eq!(
        forward_output.len(),
        numel,
        "cumprod forward output length must equal input numel"
    );

    // Backward needs grad_output (use the forward output again as
    // a representative tensor), input, output. All same shape.
    let grad_output = forward_output.clone();
    let backward = match cumprod_backward_tensor_contiguous_f64(
        &grad_output,
        &storage,
        &forward_output,
        &meta,
        dim,
    ) {
        Ok(g) => g,
        Err(_) => return,
    };
    assert_eq!(
        backward.len(),
        numel,
        "cumprod backward output length must equal input numel"
    );
});
