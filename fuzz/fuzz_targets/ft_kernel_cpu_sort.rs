#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::sort_tensor_contiguous_f64;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // Preamble: ndim, dim, descending flag, then shape bytes.
    // The descending bit doubles the effective coverage: any
    // bug in the lane-buffer reuse between ascending and
    // descending paths would surface across runs.
    let ndim = usize::from(data[0] % 7);
    if ndim == 0 {
        return;
    }
    let dim = usize::from(data[1] % (ndim as u8).max(1)) % ndim;
    let descending = (data[2] & 1) == 1;
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

    // Generate a mix of finite, NaN, and ±inf values so the
    // partial_cmp -> Ordering::Equal NaN-fallback path is
    // exercised.
    let storage: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(ndim + i) % body.len()];
            match raw % 16 {
                // 1/16 odds: NaN — exercises partial_cmp's None path.
                0 => f64::NAN,
                // 1/16: +inf
                1 => f64::INFINITY,
                // 1/16: -inf
                2 => f64::NEG_INFINITY,
                _ => f64::from(raw as i8) / 11.0,
            }
        })
        .collect();

    let (sorted_values, indices) =
        match sort_tensor_contiguous_f64(&storage, &meta, dim, descending) {
            Ok(pair) => pair,
            Err(_) => return,
        };

    // Both outputs must equal numel.
    assert_eq!(
        sorted_values.len(),
        numel,
        "sort sorted_values length must equal numel"
    );
    assert_eq!(indices.len(), numel, "sort indices length must equal numel");

    // Per-lane reconstruction: input[indices[i]] must bit-equal
    // sorted_values[i]. The lane is the (outer, *, inner) slice;
    // recompute the source linear index from the kernel's stride
    // formula.
    let dim_size = shape[dim];
    if dim_size == 0 {
        return;
    }
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut seen = vec![false; dim_size];
            for d in 0..dim_size {
                let out_idx = outer * dim_size * inner_size + d * inner_size + inner;
                let orig_d = indices[out_idx];
                // (1) Index validity: every per-lane index is in
                // [0, dim_size).
                assert!(
                    orig_d < dim_size,
                    "sort index out of range: orig_d={orig_d} dim_size={dim_size}"
                );
                // (2) Permutation property: each index in [0,
                // dim_size) appears exactly once per lane.
                assert!(
                    !seen[orig_d],
                    "sort index appears twice in lane: orig_d={orig_d} (outer={outer}, inner={inner})"
                );
                seen[orig_d] = true;
                // (3) Reconstruction: input[indices[i]] bit-equals
                // sorted_values[i]. NaN compares as not-equal-to-
                // self under == so use to_bits for the equality
                // check, which gives bit-identity.
                let src_idx = outer * dim_size * inner_size + orig_d * inner_size + inner;
                assert_eq!(
                    storage[src_idx].to_bits(),
                    sorted_values[out_idx].to_bits(),
                    "sort reconstruction mismatch: input[{src_idx}] != sorted_values[{out_idx}]"
                );
            }
        }
    }
});
