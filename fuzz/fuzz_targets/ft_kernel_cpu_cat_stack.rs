#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{cat_tensor_contiguous_f64, stack_tensor_contiguous_f64};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 6;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let ndim = usize::from(data[0] % 4); // 1..=3
    if ndim == 0 {
        return;
    }
    let n_inputs = usize::from(2 + (data[1] % 3)); // 2..=4
    let cat_dim = usize::from(data[2] % (ndim as u8).max(1)) % ndim;
    // stack_dim can be 0..=ndim (inserts a new dim).
    let stack_dim = usize::from(data[3] % (ndim as u8 + 1)) % (ndim + 1);
    let body = &data[4..];

    if body.len() < ndim + n_inputs {
        return;
    }

    // Build base shape from first ndim body bytes.
    let base_shape: Vec<usize> = body[..ndim]
        .iter()
        .map(|b| usize::from(b % (MAX_SHAPE_DIM + 1)))
        .collect();

    // Each input has the same base shape except cat_dim can vary.
    // (For stack all shapes must match exactly.) Build per-input
    // cat shapes by perturbing base_shape[cat_dim] from body bytes.
    let cat_shapes: Vec<Vec<usize>> = (0..n_inputs)
        .map(|i| {
            let mut s = base_shape.clone();
            s[cat_dim] = usize::from(body[ndim + i] % (MAX_SHAPE_DIM + 1));
            s
        })
        .collect();

    // Build cat inputs. Skip if any shape volume overflows our cap.
    let mut cat_storages: Vec<Vec<f64>> = Vec::with_capacity(n_inputs);
    let mut cat_metas: Vec<TensorMeta> = Vec::with_capacity(n_inputs);
    for shape in &cat_shapes {
        let meta = match TensorMeta::from_shape_and_strides(
            shape.clone(),
            ft_core::contiguous_strides(shape),
            0,
            DType::F64,
            Device::Cpu,
        ) {
            Ok(m) => m,
            Err(_) => return,
        };
        let numel = meta.numel();
        if numel > 4096 {
            return;
        }
        let storage: Vec<f64> = (0..numel)
            .map(|i| {
                let raw = body.get(i % body.len().max(1)).copied().unwrap_or(0) as i32;
                f64::from(raw - 128) / 40.0
            })
            .collect();
        cat_storages.push(storage);
        cat_metas.push(meta);
    }

    // Drive cat. Total cat-dim size and expected output length.
    let cat_inputs: Vec<(&[f64], &TensorMeta)> = cat_storages
        .iter()
        .zip(cat_metas.iter())
        .map(|(s, m)| (s.as_slice(), m))
        .collect();
    if let Ok(out) = cat_tensor_contiguous_f64(&cat_inputs, cat_dim) {
        let expected_numel: usize = cat_storages.iter().map(|s| s.len()).sum();
        assert_eq!(
            out.len(),
            expected_numel,
            "cat output length must equal sum of input numel"
        );
    }

    // Drive stack with the BASE shape (all inputs identical). This
    // is the only shape combination that stack accepts.
    let stack_meta = match TensorMeta::from_shape_and_strides(
        base_shape.clone(),
        ft_core::contiguous_strides(&base_shape),
        0,
        DType::F64,
        Device::Cpu,
    ) {
        Ok(m) => m,
        Err(_) => return,
    };
    let per_input_numel = stack_meta.numel();
    if per_input_numel > 4096 {
        return;
    }
    let stack_storage: Vec<f64> = (0..per_input_numel)
        .map(|i| {
            let raw = body.get(i % body.len().max(1)).copied().unwrap_or(0) as i32;
            f64::from(raw - 128) / 40.0
        })
        .collect();
    let stack_inputs: Vec<(&[f64], &TensorMeta)> = (0..n_inputs)
        .map(|_| (stack_storage.as_slice(), &stack_meta))
        .collect();
    if let Ok(out) = stack_tensor_contiguous_f64(&stack_inputs, stack_dim) {
        let expected = per_input_numel * n_inputs;
        assert_eq!(
            out.len(),
            expected,
            "stack output length must equal n_inputs * per_input_numel"
        );
    }
});
