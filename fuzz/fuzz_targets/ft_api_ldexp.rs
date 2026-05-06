#![no_main]

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 192;
const MAX_RANK: usize = 3;
const MAX_DIM: u8 = 4;
const MAX_NUMEL: usize = 64;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let data = &data[..data.len().min(MAX_INPUT_BYTES)];
    let shape_header = data[0];
    let mode_header = data.get(1).copied().unwrap_or(0);

    let input_rank = usize::from(shape_header % (MAX_RANK as u8 + 1));
    let input_shape_start = 2;
    let input_shape = decode_shape(data, input_shape_start, input_rank, 0x31);
    let input_numel = match checked_numel(input_shape.as_slice()) {
        Some(numel) if numel <= MAX_NUMEL => numel,
        _ => return,
    };

    let input_values_start = input_shape_start + input_rank;
    let other_shape_same_as_input = mode_header & 0x80 == 0;
    let other_rank = if other_shape_same_as_input {
        input_rank
    } else {
        usize::from((mode_header >> 4) % (MAX_RANK as u8 + 1))
    };
    let other_shape_start = input_values_start + input_numel;
    let other_shape = if other_shape_same_as_input {
        input_shape.clone()
    } else {
        decode_shape(data, other_shape_start, other_rank, 0x73)
    };
    let other_numel = match checked_numel(other_shape.as_slice()) {
        Some(numel) if numel <= MAX_NUMEL => numel,
        _ => return,
    };

    let other_values_start =
        other_shape_start + usize::from(!other_shape_same_as_input) * other_rank;
    let input_values = decode_values(data, input_values_start, input_numel, decode_input_value);
    let other_values = decode_values(data, other_values_start, other_numel, decode_exponent_value);

    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let input = match session.tensor_variable(input_values.clone(), input_shape.clone(), false) {
        Ok(input) => input,
        Err(_) => return,
    };
    let other = match session.tensor_variable(other_values.clone(), other_shape.clone(), false) {
        Ok(other) => other,
        Err(_) => return,
    };

    let result = session.tensor_ldexp(input, other);
    if input_shape != other_shape {
        assert!(
            result.is_err(),
            "ldexp must reject non-broadcast mismatched shapes"
        );
        return;
    }

    let output = match result {
        Ok(output) => output,
        Err(error) => panic!("same-shape ldexp should not fail: {error:?}"),
    };
    let Ok((output_values, output_meta)) = session.tensor_values_meta(output) else {
        panic!("ldexp output must remain readable after success");
    };
    assert_eq!(
        output_meta.shape(),
        input_shape.as_slice(),
        "ldexp success must preserve input shape"
    );
    assert_eq!(
        output_values.len(),
        input_numel,
        "ldexp success must preserve logical element count"
    );

    if other_values.iter().all(|value| *value == 0.0) {
        for (actual, expected) in output_values.iter().zip(input_values.iter()) {
            assert!(
                same_identity_value(*actual, *expected),
                "ldexp(x, 0) must preserve x: got {actual:?}, expected {expected:?}"
            );
        }
    }

    for ((actual, x), n) in output_values
        .iter()
        .zip(input_values.iter())
        .zip(other_values.iter())
    {
        if x.is_finite() && n.is_finite() && n.abs() <= 16.0 {
            let expected = *x * 2.0_f64.powf(*n);
            assert!(
                close_enough(*actual, expected),
                "ldexp({x:?}, {n:?}) = {actual:?}, expected {expected:?}"
            );
        }
    }
});

fn decode_shape(data: &[u8], start: usize, rank: usize, salt: u8) -> Vec<usize> {
    (0..rank)
        .map(|index| {
            let byte = data
                .get(start + index)
                .copied()
                .unwrap_or_else(|| salt.wrapping_add(index as u8));
            usize::from(byte % (MAX_DIM + 1))
        })
        .collect()
}

fn checked_numel(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1_usize, |acc, dim| acc.checked_mul(*dim))
}

fn decode_values(data: &[u8], start: usize, len: usize, decode: fn(u8) -> f64) -> Vec<f64> {
    (0..len)
        .map(|index| {
            let byte = data
                .get(start + index)
                .copied()
                .unwrap_or_else(|| (index as u8).wrapping_mul(37).wrapping_add(11));
            decode(byte)
        })
        .collect()
}

fn decode_input_value(byte: u8) -> f64 {
    match byte % 18 {
        0 => f64::NAN,
        1 => f64::INFINITY,
        2 => f64::NEG_INFINITY,
        3 => -0.0,
        4 => 0.0,
        5 => 1.0,
        6 => -1.0,
        7 => 0.5,
        8 => -0.5,
        9 => 1024.0,
        10 => -1024.0,
        11 => f64::MIN_POSITIVE,
        12 => -f64::MIN_POSITIVE,
        _ => (f64::from(byte) - 128.0) / 5.0,
    }
}

fn decode_exponent_value(byte: u8) -> f64 {
    match byte % 20 {
        0 => 0.0,
        1 => 1.0,
        2 => -1.0,
        3 => 2.0,
        4 => -2.0,
        5 => 10.0,
        6 => -10.0,
        7 => 63.0,
        8 => -63.0,
        9 => 1024.0,
        10 => -1075.0,
        11 => f64::INFINITY,
        12 => f64::NEG_INFINITY,
        13 => f64::NAN,
        _ => f64::from(i8::from_ne_bytes([byte])) / 4.0,
    }
}

fn same_identity_value(actual: f64, expected: f64) -> bool {
    if expected.is_nan() {
        actual.is_nan()
    } else {
        actual.to_bits() == expected.to_bits()
    }
}

fn close_enough(actual: f64, expected: f64) -> bool {
    if actual.is_nan() && expected.is_nan() {
        return true;
    }
    if actual.is_infinite() || expected.is_infinite() {
        return actual == expected;
    }
    if actual.to_bits() == expected.to_bits() {
        return true;
    }
    let scale = actual.abs().max(expected.abs()).max(1.0);
    (actual - expected).abs() <= 64.0 * scale * f64::EPSILON
}
