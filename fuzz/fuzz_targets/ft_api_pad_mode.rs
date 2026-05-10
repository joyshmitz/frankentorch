#![no_main]

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 256;
const MAX_RANK: usize = 4;
const MAX_DIM: u8 = 4;
const MAX_PAD: u8 = 3;
const MAX_INPUT_NUMEL: usize = 128;
const MAX_OUTPUT_NUMEL: usize = 512;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let data = &data[..data.len().min(MAX_INPUT_BYTES)];

    let rank = 1 + usize::from(data[0] % MAX_RANK as u8);
    let mode = decode_mode(data[1]);
    let pad_pairs = usize::from(data[2] % (rank as u8 + 2));
    let odd_padding = data.get(3).is_some_and(|byte| byte & 0x80 != 0);
    let shape_start = 4;
    let padding_start = shape_start + rank;
    let value_start = padding_start + pad_pairs * 2 + usize::from(odd_padding);

    let shape = decode_shape(data, shape_start, rank);
    let input_numel = match checked_numel(shape.as_slice()) {
        Some(numel) if numel <= MAX_INPUT_NUMEL => numel,
        _ => return,
    };

    let mut padding = decode_padding(data, padding_start, pad_pairs);
    if odd_padding {
        padding.push(usize::from(
            data.get(padding_start + pad_pairs * 2)
                .copied()
                .unwrap_or(1)
                % (MAX_PAD + 1),
        ));
    }

    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let input_values = decode_values(data, value_start, input_numel);
    let input = match session.tensor_variable(input_values.clone(), shape.clone(), false) {
        Ok(input) => input,
        Err(_) => return,
    };
    let pad_value = decode_value(data.get(value_start + input_numel).copied().unwrap_or(0x55));

    let result = session.tensor_pad_mode(input, padding.as_slice(), mode, pad_value);
    let known_mode = matches!(mode, "constant" | "reflect" | "replicate" | "circular");
    if !known_mode || odd_padding || pad_pairs > rank {
        assert!(
            result.is_err(),
            "pad_mode must reject invalid mode={mode:?}, odd_padding={odd_padding}, pad_pairs={pad_pairs}, rank={rank}"
        );
        return;
    }

    let Some(expected_shape) = expected_output_shape(shape.as_slice(), padding.as_slice()) else {
        return;
    };
    let expected_numel = match checked_numel(expected_shape.as_slice()) {
        Some(numel) if numel <= MAX_OUTPUT_NUMEL => numel,
        _ => return,
    };

    let output = match result {
        Ok(output) => output,
        Err(error) => panic!("valid pad_mode input failed: mode={mode}, error={error:?}"),
    };
    let Ok((output_values, output_meta)) = session.tensor_values_meta(output) else {
        panic!("pad_mode output must remain readable after success");
    };
    assert_eq!(
        output_meta.shape(),
        expected_shape.as_slice(),
        "pad_mode success must match padding-derived output shape"
    );
    assert_eq!(
        output_values.len(),
        expected_numel,
        "pad_mode success must preserve output logical element count"
    );

    if padding.iter().all(|pad| *pad == 0) {
        assert_eq!(
            output_values.len(),
            input_values.len(),
            "zero padding must preserve input length"
        );
        for (actual, expected) in output_values.iter().zip(input_values.iter()) {
            assert!(
                same_value(*actual, *expected),
                "zero padding in mode={mode} must preserve values: got {actual:?}, expected {expected:?}"
            );
        }
    }
});

fn decode_mode(selector: u8) -> &'static str {
    match selector % 5 {
        0 => "constant",
        1 => "reflect",
        2 => "replicate",
        3 => "circular",
        _ => "invalid",
    }
}

fn decode_shape(data: &[u8], start: usize, rank: usize) -> Vec<usize> {
    (0..rank)
        .map(|index| {
            let byte = data
                .get(start + index)
                .copied()
                .unwrap_or_else(|| (index as u8).wrapping_mul(29).wrapping_add(1));
            1 + usize::from(byte % MAX_DIM)
        })
        .collect()
}

fn decode_padding(data: &[u8], start: usize, pairs: usize) -> Vec<usize> {
    (0..pairs * 2)
        .map(|index| {
            let byte = data
                .get(start + index)
                .copied()
                .unwrap_or_else(|| (index as u8).wrapping_mul(17).wrapping_add(3));
            usize::from(byte % (MAX_PAD + 1))
        })
        .collect()
}

fn decode_values(data: &[u8], start: usize, len: usize) -> Vec<f64> {
    (0..len)
        .map(|index| {
            let byte = data
                .get(start + index)
                .copied()
                .unwrap_or_else(|| (index as u8).wrapping_mul(37).wrapping_add(11));
            decode_value(byte)
        })
        .collect()
}

fn decode_value(byte: u8) -> f64 {
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
        9 => 2.0,
        10 => -2.0,
        _ => (f64::from(byte) - 128.0) / 7.0,
    }
}

fn expected_output_shape(shape: &[usize], padding: &[usize]) -> Option<Vec<usize>> {
    let mut out_shape = shape.to_vec();
    for (pair, pads) in padding.chunks_exact(2).enumerate() {
        let dim = shape.len() - 1 - pair;
        let pad_total = pads
            .iter()
            .try_fold(0_usize, |acc, pad| acc.checked_add(*pad))?;
        let slot = out_shape.get_mut(dim)?;
        *slot = slot.checked_add(pad_total)?;
    }
    Some(out_shape)
}

fn checked_numel(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1_usize, |acc, dim| acc.checked_mul(*dim))
}

fn same_value(actual: f64, expected: f64) -> bool {
    if expected.is_nan() {
        actual.is_nan()
    } else {
        actual.to_bits() == expected.to_bits()
    }
}
