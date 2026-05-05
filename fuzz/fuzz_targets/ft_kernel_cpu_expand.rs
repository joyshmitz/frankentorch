#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::expand_tensor_contiguous_f64;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 256;

// Cap each shape dim at 8 to keep numel tractable for the fuzzer
// (worst-case rank-8 = 8^8 ≈ 16M; capped target_numel below at
// 4096 to keep iteration fast). The kernel's checked arithmetic
// should reject any overflow without panicking; we want the
// fuzzer iterating on inputs the validator actually walks.
const MAX_SHAPE_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // Preamble: 1 byte input rank, 1 byte target rank.
    let input_rank = usize::from(data[0] % 9);
    let target_rank = usize::from(data[1] % 9);
    let body = &data[2..];

    if body.len() < input_rank + target_rank {
        return;
    }
    let input_shape: Vec<usize> = body[..input_rank]
        .iter()
        .map(|b| usize::from(b % (MAX_SHAPE_DIM + 1)))
        .collect();
    let target_shape: Vec<usize> = body[input_rank..input_rank + target_rank]
        .iter()
        .map(|b| usize::from(b % (MAX_SHAPE_DIM + 1)))
        .collect();

    // Build contiguous F64 meta. Skip if construction fails (e.g.
    // some pathological dim-0 layouts the validator rejects
    // ahead-of-time, no need to fuzz those further).
    let meta = match TensorMeta::from_shape_and_strides(
        input_shape.clone(),
        ft_core::contiguous_strides(&input_shape),
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
    // Cap target_numel similarly to keep iteration fast.
    let target_numel: usize = target_shape.iter().product();
    if target_numel > 4096 {
        return;
    }
    let storage = vec![0.0_f64; numel];

    if let Ok(output) = expand_tensor_contiguous_f64(&storage, &meta, &target_shape) {
        // On success, output length must equal product(target_shape).
        // Recompute independently to catch any allocation arithmetic
        // off-by-one in the kernel.
        let expected: usize = target_shape.iter().product();
        assert_eq!(
            output.len(),
            expected,
            "expand output length must equal product(target_shape)"
        );
    }
});
