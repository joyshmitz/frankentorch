#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::index_put_tensor_contiguous_f64;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 6;

fuzz_target!(|data: &[u8]| {
    if data.len() < 5 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let ndim = usize::from(1 + (data[0] % 3)); // 1..=3
    let num_indexed_dims = usize::from(1 + (data[1] % ndim as u8)).min(ndim);
    let n_indices = usize::from(1 + (data[2] % 8)); // 1..=8 index entries
    let accumulate = data[3] & 1 == 1;
    let body = &data[4..];

    if body.len() < ndim {
        return;
    }
    let shape: Vec<usize> = body[..ndim]
        .iter()
        .map(|b| usize::from(1 + (b % MAX_SHAPE_DIM))) // 1..=6 — no zero extents
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

    // Suffix size: product of non-indexed dims.
    let suffix_size: usize = shape[num_indexed_dims..].iter().product();
    let values_needed = n_indices * suffix_size;
    if values_needed > 4096 {
        return;
    }

    let input: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.001).collect();

    // Build one index tensor per indexed dim, with each index
    // value in [0, shape[d]).
    let indices: Vec<Vec<f64>> = (0..num_indexed_dims)
        .map(|d| {
            (0..n_indices)
                .map(|i| {
                    let raw = body[(ndim + d * n_indices + i) % body.len().max(1)];
                    (usize::from(raw) % shape[d]) as f64
                })
                .collect()
        })
        .collect();

    // Distinguishable values so accumulate vs overwrite shows up.
    let values: Vec<f64> = (0..values_needed).map(|i| 100.0 + (i as f64)).collect();

    let output = match index_put_tensor_contiguous_f64(&input, &meta, &indices, &values, accumulate)
    {
        Ok(out) => out,
        Err(_) => return,
    };
    assert_eq!(output.len(), numel, "index_put output length");

    // For overwrite mode (accumulate=false), the cells written by
    // index_put must EXACTLY equal the corresponding values entry
    // (the original input value is overwritten). The remaining
    // cells (those not addressed by any index tuple) must retain
    // their input value.
    if !accumulate {
        // Compute the set of flat positions that were written.
        let mut indexed_strides = vec![0usize; num_indexed_dims];
        for d in 0..num_indexed_dims {
            indexed_strides[d] = shape[d + 1..].iter().product();
        }
        // Walk every (i, s) pair and verify output[base + s] == values[i * suffix_size + s].
        for i in 0..n_indices {
            let mut base = 0usize;
            for d in 0..num_indexed_dims {
                base += (indices[d][i] as usize) * indexed_strides[d];
            }
            for s in 0..suffix_size {
                let got = output[base + s];
                let expected = values[i * suffix_size + s];
                // Use bit-equal because overwrite is a direct copy.
                assert!(
                    got.to_bits() == expected.to_bits(),
                    "index_put(overwrite)[base={base}, s={s}] = {got}, expected = {expected}"
                );
            }
        }
    }

    // For accumulate mode: every written cell must satisfy
    // output[p] >= input[p] (or <= input[p] for negative values).
    // Since values are positive (100.0+), output must be >= input
    // at every written position.
    if accumulate {
        let mut indexed_strides = vec![0usize; num_indexed_dims];
        for d in 0..num_indexed_dims {
            indexed_strides[d] = shape[d + 1..].iter().product();
        }
        for i in 0..n_indices {
            let mut base = 0usize;
            for d in 0..num_indexed_dims {
                base += (indices[d][i] as usize) * indexed_strides[d];
            }
            for s in 0..suffix_size {
                let got = output[base + s];
                let original = input[base + s];
                // Output is original + accumulated values; values are all positive
                // so output must be >= original.
                assert!(
                    got >= original - 1e-12,
                    "index_put(accumulate)[base={base}, s={s}] = {got} < original {original} despite positive values"
                );
            }
        }
    }
});
