#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::masked_fill_tensor_contiguous_f64;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // Preamble: ndim, length_mode (0 = match, 1 = truncated mask),
    // value_bits (0 = finite, 1 = NaN), then shape bytes.
    let ndim = usize::from(data[0] % 7);
    if ndim == 0 {
        return;
    }
    let length_mode = data[1] % 2;
    let value_nan = (data[2] & 1) == 1;
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

    // Input mixes finite, NaN, ±inf so the kernel passes
    // non-replaced cells through bit-exactly even for specials.
    let input: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(ndim + i) % body.len()];
            match raw % 8 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                _ => f64::from(raw as i8) / 7.0,
            }
        })
        .collect();

    // Mask values: 0.0 / 1.0 / 2.0 (any non-zero counts as fill).
    let mask: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(2 * ndim + i) % body.len()];
            match raw % 4 {
                0 => 0.0,
                1 => 1.0,
                _ => 2.5, // exercise the "any non-zero" branch
            }
        })
        .collect();

    let mask_slice: &[f64] = match length_mode {
        // All-match: kernel must succeed.
        0 => &mask[..],
        // Truncated: kernel must reject with ShapeMismatch.
        1 if numel >= 1 => &mask[..numel - 1],
        _ => return,
    };

    let value = if value_nan { f64::NAN } else { 42.0_f64 };

    match masked_fill_tensor_contiguous_f64(&input, &meta, mask_slice, value) {
        Ok(output) => {
            assert_eq!(
                output.len(),
                numel,
                "masked_fill output length must equal numel"
            );
            for (i, out) in output.iter().enumerate() {
                let m = mask[i];
                if m != 0.0 {
                    // Fill: output bit-equals value.
                    assert_eq!(
                        out.to_bits(),
                        value.to_bits(),
                        "masked_fill: cell {i} mask={m} should be value but got {out}"
                    );
                } else {
                    // Pass-through: output bit-equals input[i].
                    assert_eq!(
                        out.to_bits(),
                        input[i].to_bits(),
                        "masked_fill: cell {i} mask=0 should be input but got {out}"
                    );
                }
            }
        }
        Err(_) => {
            // Truncated mask path returns Err — fine; contract is
            // no panic.
        }
    }
});
