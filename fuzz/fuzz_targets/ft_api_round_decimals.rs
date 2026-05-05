#![no_main]

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 192;
const MAX_RANK: usize = 3;
const MAX_DIM: u8 = 4;
const MAX_NUMEL: usize = 64;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let rank = usize::from(data[0] % (MAX_RANK as u8 + 1));
    let shape_start = 2;
    let values_start = shape_start + rank;
    if data.len() < values_start {
        return;
    }

    let decimals = decode_decimals(data[1], data.get(values_start).copied().unwrap_or(0));
    let shape: Vec<usize> = data[shape_start..values_start]
        .iter()
        .map(|byte| usize::from(byte % (MAX_DIM + 1)))
        .collect();
    let numel = match checked_numel(&shape) {
        Some(numel) if numel <= MAX_NUMEL => numel,
        _ => return,
    };

    let mut values = Vec::with_capacity(numel);
    for index in 0..numel {
        let byte = data
            .get(values_start + index)
            .copied()
            .unwrap_or_else(|| (index as u8).wrapping_mul(37));
        values.push(decode_value(byte));
    }

    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let input = match session.tensor_variable(values, shape.clone(), false) {
        Ok(input) => input,
        Err(_) => return,
    };

    if let Ok(output) = session.tensor_round_decimals(input, decimals) {
        let Ok((output_values, output_meta)) = session.tensor_values_meta(output) else {
            panic!("round_decimals output must remain readable after success");
        };
        assert_eq!(
            output_meta.shape(),
            shape.as_slice(),
            "round_decimals success must preserve input shape"
        );
        assert_eq!(
            output_values.len(),
            numel,
            "round_decimals success must preserve logical element count"
        );
        if decimals == 0 {
            for value in output_values {
                assert!(
                    value.is_nan() || value.fract() == 0.0,
                    "decimals=0 output must be integral or NaN"
                );
            }
        }
    }
});

fn checked_numel(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1_usize, |acc, dim| acc.checked_mul(*dim))
}

fn decode_decimals(selector: u8, extra: u8) -> i32 {
    match selector % 16 {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => -1,
        4 => -2,
        5 => 6,
        6 => -6,
        7 => 20,
        8 => -20,
        9 => 308,
        10 => -308,
        11 => 309,
        12 => -324,
        _ => i32::from(i8::from_ne_bytes([selector])) + i32::from(extra % 17) - 8,
    }
}

fn decode_value(byte: u8) -> f64 {
    match byte % 16 {
        0 => f64::NAN,
        1 => f64::INFINITY,
        2 => f64::NEG_INFINITY,
        3 => -0.0,
        4 => 0.0,
        5 => 0.5,
        6 => -0.5,
        7 => 1.2345,
        8 => -1.2345,
        9 => 1234.5,
        10 => -1267.0,
        _ => (f64::from(byte) - 128.0) / 3.0,
    }
}
