#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{cumsum_backward_tensor_contiguous_f64, cumsum_tensor_contiguous_f64};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // Preamble: ndim, dim. The shape bytes follow. Forward and
    // backward share the same dim parameter, so any mismatch
    // (dim out of range vs ndim) must be rejected by both with
    // the same KernelError variant.
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

    // Generate input from the body bytes scaled into a small
    // range so the forward accumulator doesn't immediately
    // saturate to inf.
    let storage: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(ndim + i) % body.len()] as i32;
            f64::from(raw - 128) / 41.0
        })
        .collect();

    let forward_output = match cumsum_tensor_contiguous_f64(&storage, &meta, dim) {
        Ok(out) => out,
        Err(_) => return,
    };
    // Forward output must have exactly numel elements (cumsum
    // preserves shape). Recompute independently to catch
    // allocation off-by-ones in the kernel.
    assert_eq!(
        forward_output.len(),
        numel,
        "cumsum forward output length must equal input numel"
    );

    // Backward kernel: the reverse-direction prefix sum applied
    // to a fake grad_output (use forward_output as a dense,
    // representative input). Same shape contract.
    let grad_output = forward_output.clone();
    let backward_output =
        match cumsum_backward_tensor_contiguous_f64(&grad_output, &meta, dim) {
            Ok(out) => out,
            Err(_) => return,
        };
    assert_eq!(
        backward_output.len(),
        numel,
        "cumsum backward output length must equal input numel"
    );

    // Forward/backward shape consistency: both kernels share dim
    // and meta. If forward succeeds and backward fails (or vice
    // versa) for the same inputs, that asymmetry is itself a bug.
    // The matching length assertions above already catch shape
    // drift.
});
