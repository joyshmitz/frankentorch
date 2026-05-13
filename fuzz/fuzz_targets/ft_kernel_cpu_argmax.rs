#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{
    argmax_dim_tensor_contiguous_f64, argmin_dim_tensor_contiguous_f64,
};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let ndim = usize::from(data[0] % 6);
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

    let storage: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(ndim + i) % body.len()] as i32;
            f64::from(raw - 128) / 30.0
        })
        .collect();

    // Skip the empty-output / zero-reduce case to keep the
    // invariants well-defined.
    if numel == 0 {
        return;
    }
    let reduce_size = shape[dim];
    if reduce_size == 0 {
        return;
    }
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

    let argmax = match argmax_dim_tensor_contiguous_f64(&storage, &meta, dim) {
        Ok(v) => v,
        Err(_) => return,
    };
    let argmin = match argmin_dim_tensor_contiguous_f64(&storage, &meta, dim) {
        Ok(v) => v,
        Err(_) => return,
    };
    assert_eq!(argmax.len(), out_numel, "argmax length");
    assert_eq!(argmin.len(), out_numel, "argmin length");

    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    // Index correctness: for every output slot, the returned index
    // must point to the actual max/min along that slice. Skip
    // slices containing any NaN — in that case the kernel picks
    // the first NaN position, which is harder to validate without
    // mirroring the kernel's NaN-handling exactly.
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let flat_out = outer * inner_size + inner;
            let max_idx_f = argmax[flat_out];
            let min_idx_f = argmin[flat_out];
            let max_idx = max_idx_f as usize;
            let min_idx = min_idx_f as usize;
            assert!(
                (max_idx_f - max_idx as f64).abs() < f64::EPSILON,
                "argmax not integer: {max_idx_f}"
            );
            assert!(
                (min_idx_f - min_idx as f64).abs() < f64::EPSILON,
                "argmin not integer: {min_idx_f}"
            );
            assert!(
                max_idx < reduce_size,
                "argmax {max_idx} >= reduce_size {reduce_size}"
            );
            assert!(
                min_idx < reduce_size,
                "argmin {min_idx} >= reduce_size {reduce_size}"
            );

            // Walk the slice. If any cell is NaN, the kernel
            // contract is "first NaN wins" — skip validation.
            let mut any_nan = false;
            let mut slice_max = f64::NEG_INFINITY;
            let mut slice_min = f64::INFINITY;
            for r in 0..reduce_size {
                let v = storage[outer * reduce_size * inner_size + r * inner_size + inner];
                if v.is_nan() {
                    any_nan = true;
                    break;
                }
                if v > slice_max {
                    slice_max = v;
                }
                if v < slice_min {
                    slice_min = v;
                }
            }
            if any_nan {
                continue;
            }

            let selected_max =
                storage[outer * reduce_size * inner_size + max_idx * inner_size + inner];
            let selected_min =
                storage[outer * reduce_size * inner_size + min_idx * inner_size + inner];
            assert!(
                selected_max == slice_max,
                "argmax[{outer},{inner}] = {max_idx} -> val {selected_max}, but slice max is {slice_max}"
            );
            assert!(
                selected_min == slice_min,
                "argmin[{outer},{inner}] = {min_idx} -> val {selected_min}, but slice min is {slice_min}"
            );
        }
    }
});
