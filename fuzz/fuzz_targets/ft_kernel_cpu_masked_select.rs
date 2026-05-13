#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::masked_select_tensor_contiguous_f64;
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
    let length_mode = data[1] % 2;
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

    let input: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(ndim + i) % body.len()];
            match raw % 8 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                _ => f64::from(raw as i8) / 11.0,
            }
        })
        .collect();

    // Non-boolean mask values (the kernel treats any non-zero as
    // "select").
    let mask: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(2 * ndim + i) % body.len()];
            match raw % 4 {
                0 => 0.0,
                1 => 1.0,
                2 => -3.7,
                _ => 99.0,
            }
        })
        .collect();

    let mask_slice: &[f64] = match length_mode {
        0 => &mask[..],
        1 if numel >= 1 => &mask[..numel - 1],
        _ => return,
    };

    match masked_select_tensor_contiguous_f64(&input, mask_slice, &meta) {
        Ok(output) => {
            // Output length must equal count of non-zero mask cells.
            let expected_len = mask.iter().filter(|m| **m != 0.0).count();
            assert_eq!(
                output.len(),
                expected_len,
                "masked_select output length mismatch"
            );

            // Every output element must bit-equal some input cell
            // where the corresponding mask was non-zero (preserves
            // order). Walk both in tandem.
            let mut out_iter = output.iter();
            for (i, m) in mask.iter().enumerate() {
                if *m != 0.0 {
                    let next = out_iter
                        .next()
                        .expect("output exhausted before mask non-zero count");
                    assert_eq!(
                        next.to_bits(),
                        input[i].to_bits(),
                        "masked_select cell {i}: expected input bits, got {next}"
                    );
                }
            }
            assert!(out_iter.next().is_none(), "masked_select trailing element");
        }
        Err(_) => {
            // Truncated mask path returns Err — fine.
        }
    }
});
