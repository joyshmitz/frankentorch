#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{
    addmm_tensor_contiguous_f32, addmv_tensor_contiguous_f32,
    bmm_tensor_contiguous_f32, cumprod_backward_tensor_contiguous_f32,
    cumprod_tensor_contiguous_f32, scatter_add_tensor_contiguous_f32,
};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_DIM: u8 = 8;

fn build_meta(shape: &[usize], dtype: DType) -> Option<TensorMeta> {
    TensorMeta::from_shape_and_strides(
        shape.to_vec(),
        ft_core::contiguous_strides(shape),
        0,
        dtype,
        Device::Cpu,
    )
    .ok()
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let batch = usize::from(data[0] % (MAX_DIM + 1));
    let m = usize::from(data[1] % (MAX_DIM + 1));
    let k = usize::from(data[2] % (MAX_DIM + 1));
    let n = usize::from(data[3] % (MAX_DIM + 1));
    let alpha = (data[4] as i32 - 128) as f32 / 64.0;
    let beta = (data[5] as i32 - 128) as f32 / 64.0;
    let body = &data[6..];

    // --- bmm f32 ---
    if batch * m * k <= 4096 && batch * k * n <= 4096 && batch * m * n <= 4096 {
        let lhs_shape = vec![batch, m, k];
        let rhs_shape = vec![batch, k, n];
        if let (Some(lhs_meta), Some(rhs_meta)) =
            (build_meta(&lhs_shape, DType::F32), build_meta(&rhs_shape, DType::F32))
        {
            let lhs_numel = batch * m * k;
            let rhs_numel = batch * k * n;
            let lhs: Vec<f32> = (0..lhs_numel)
                .map(|i| {
                    let raw = body.get(i % body.len().max(1)).copied().unwrap_or(0) as i32;
                    (raw - 128) as f32 / 40.0
                })
                .collect();
            let rhs: Vec<f32> = (0..rhs_numel)
                .map(|i| {
                    let raw = body.get((lhs_numel + i) % body.len().max(1)).copied().unwrap_or(0) as i32;
                    (raw - 128) as f32 / 40.0
                })
                .collect();
            if let Ok(out) = bmm_tensor_contiguous_f32(&lhs, &rhs, &lhs_meta, &rhs_meta) {
                assert_eq!(out.len(), batch * m * n, "bmm_f32 length");
                for (i, &v) in out.iter().enumerate() {
                    assert!(v.is_finite(), "bmm_f32[{i}] = {v} non-finite");
                }
            }
        }
    }

    // --- addmm f32 (with bias 2D) ---
    if m * k <= 4096 && k * n <= 4096 && m * n <= 4096 && m > 0 && k > 0 && n > 0 {
        let mat1_shape = vec![m, k];
        let mat2_shape = vec![k, n];
        let bias_shape = vec![m, n];
        if let (Some(m1m), Some(m2m), Some(bm)) = (
            build_meta(&mat1_shape, DType::F32),
            build_meta(&mat2_shape, DType::F32),
            build_meta(&bias_shape, DType::F32),
        ) {
            let mat1: Vec<f32> = (0..(m * k))
                .map(|i| {
                    let raw = body.get(i % body.len().max(1)).copied().unwrap_or(0) as i32;
                    (raw - 128) as f32 / 50.0
                })
                .collect();
            let mat2: Vec<f32> = (0..(k * n))
                .map(|i| {
                    let raw = body.get((m * k + i) % body.len().max(1)).copied().unwrap_or(0) as i32;
                    (raw - 128) as f32 / 50.0
                })
                .collect();
            let bias: Vec<f32> = vec![0.5_f32; m * n];
            if let Ok(out) =
                addmm_tensor_contiguous_f32(&bias, &mat1, &mat2, &bm, &m1m, &m2m, beta, alpha)
            {
                assert_eq!(out.len(), m * n, "addmm_f32 length");
                for (i, &v) in out.iter().enumerate() {
                    assert!(v.is_finite(), "addmm_f32[{i}] = {v} non-finite");
                }
            }
        }
    }

    // --- addmv f32 ---
    if m * k <= 4096 && m > 0 && k > 0 {
        let mat_shape = vec![m, k];
        let vec_shape = vec![k];
        let input_shape = vec![m];
        if let (Some(mm), Some(vm), Some(im)) = (
            build_meta(&mat_shape, DType::F32),
            build_meta(&vec_shape, DType::F32),
            build_meta(&input_shape, DType::F32),
        ) {
            let mat: Vec<f32> = (0..(m * k))
                .map(|i| {
                    let raw = body.get(i % body.len().max(1)).copied().unwrap_or(0) as i32;
                    (raw - 128) as f32 / 40.0
                })
                .collect();
            let v_data: Vec<f32> = (0..k)
                .map(|i| {
                    let raw = body.get((m * k + i) % body.len().max(1)).copied().unwrap_or(0) as i32;
                    (raw - 128) as f32 / 40.0
                })
                .collect();
            let bias: Vec<f32> = vec![0.5_f32; m];
            if let Ok(out) =
                addmv_tensor_contiguous_f32(&bias, &mat, &v_data, &im, &mm, &vm, beta, alpha)
            {
                assert_eq!(out.len(), m, "addmv_f32 length");
                for (i, &v) in out.iter().enumerate() {
                    assert!(v.is_finite(), "addmv_f32[{i}] = {v} non-finite");
                }
            }
        }
    }

    // --- scatter_add f32 (1-D path with dup-bias) ---
    let scatter_dim = 0usize;
    let scatter_outer = usize::from(2 + (data[0] % 4));
    let scatter_idx_count = usize::from(2 + (data[1] % 6));
    if scatter_outer > 0 && scatter_idx_count > 0 {
        let storage_shape = vec![scatter_outer];
        let idx_shape = vec![scatter_idx_count];
        if let (Some(sm), Some(im)) = (
            build_meta(&storage_shape, DType::F32),
            build_meta(&idx_shape, DType::F64),
        ) {
            let storage: Vec<f32> = (0..scatter_outer).map(|i| (i as f32) * 0.1).collect();
            // Collapse indices to {0, 1} to force accumulation.
            let index: Vec<f64> = (0..scatter_idx_count)
                .map(|i| {
                    let raw = body.get(i % body.len().max(1)).copied().unwrap_or(0);
                    (usize::from(raw) % scatter_outer.min(2).max(1)) as f64
                })
                .collect();
            let src: Vec<f32> = vec![1.0_f32; scatter_idx_count];
            if let Ok(out) =
                scatter_add_tensor_contiguous_f32(&storage, &sm, scatter_dim, &index, &im, &src)
            {
                assert_eq!(out.len(), scatter_outer, "scatter_add_f32 length");
                // Conservation: sum(out - storage) == sum(src) within f32 ULP slack.
                let delta: f32 = out.iter().zip(storage.iter()).map(|(a, b)| a - b).sum();
                let expected: f32 = src.iter().sum();
                assert!(
                    (delta - expected).abs() < 1e-3 * (1.0 + expected.abs()),
                    "scatter_add_f32 conservation: delta={delta}, expected={expected}"
                );
            }
        }
    }

    // --- cumprod f32 forward + backward ---
    let cumprod_ndim = usize::from(1 + (data[2] % 3)); // 1..=3
    if cumprod_ndim > 0 && body.len() >= cumprod_ndim {
        let cumprod_dim = usize::from(data[3] % (cumprod_ndim as u8)) % cumprod_ndim;
        let cumprod_shape: Vec<usize> = body[..cumprod_ndim]
            .iter()
            .map(|b| usize::from(b % (MAX_DIM + 1)))
            .collect();
        if let Some(cm) = build_meta(&cumprod_shape, DType::F32) {
            let cumprod_numel = cm.numel();
            if cumprod_numel > 0 && cumprod_numel <= 4096 {
                let cumprod_input: Vec<f32> = (0..cumprod_numel)
                    .map(|i| {
                        let raw = body[(cumprod_ndim + i) % body.len()];
                        match raw % 16 {
                            0 => 0.0,
                            _ => ((raw as i8) as f32 / 128.0).clamp(-1.0, 1.0),
                        }
                    })
                    .collect();
                if let Ok(cumprod_forward) =
                    cumprod_tensor_contiguous_f32(&cumprod_input, &cm, cumprod_dim)
                {
                    assert_eq!(cumprod_forward.len(), cumprod_numel, "cumprod_f32 fwd len");
                    let cumprod_grad = cumprod_forward.clone();
                    if let Ok(backward) = cumprod_backward_tensor_contiguous_f32(
                        &cumprod_grad,
                        &cumprod_input,
                        &cumprod_forward,
                        &cm,
                        cumprod_dim,
                    ) {
                        assert_eq!(backward.len(), cumprod_numel, "cumprod_f32 bwd len");
                    }
                }
            }
        }
    }
});
