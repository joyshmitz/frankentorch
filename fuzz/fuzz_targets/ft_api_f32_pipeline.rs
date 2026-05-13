#![no_main]

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 256;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let n = usize::from(2 + (data[0] % 6)); // 2..=7
    let m = usize::from(2 + (data[1] % 6)); // 2..=7
    let body = &data[2..];

    let lhs: Vec<f32> = (0..(n * m))
        .map(|i| {
            let raw = body.get(i % body.len().max(1)).copied().unwrap_or(0) as i32;
            (raw - 128) as f32 / 20.0
        })
        .collect();
    let rhs: Vec<f32> = (0..(n * m))
        .map(|i| {
            let raw = body.get((n * m + i) % body.len().max(1)).copied().unwrap_or(0) as i32;
            (raw - 128) as f32 / 20.0
        })
        .collect();

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let a = match s.tensor_variable_f32(lhs.clone(), vec![n, m], false) {
        Ok(t) => t,
        Err(_) => return,
    };
    let b = match s.tensor_variable_f32(rhs.clone(), vec![n, m], false) {
        Ok(t) => t,
        Err(_) => return,
    };

    // Chain through several elementwise ops. Each should preserve
    // F32 dtype and shape. After c9jg, these should all dispatch
    // cleanly without UnsupportedDType errors.
    let sum = match s.tensor_add(a, b) {
        Ok(t) => t,
        Err(_) => return,
    };
    let scaled = match s.tensor_mul(sum, sum) {
        Ok(t) => t,
        Err(_) => return,
    };
    let relud = match s.tensor_relu(scaled) {
        Ok(t) => t,
        Err(_) => return,
    };

    // Verify output dtype is still F32.
    let dtype = s.tensor_dtype(relud).expect("dtype");
    assert_eq!(
        dtype,
        ft_core::DType::F32,
        "F32 pipeline should preserve dtype, got {dtype:?}"
    );
    // Verify output shape unchanged.
    let shape = s.tensor_shape(relud).expect("shape");
    assert_eq!(shape, vec![n, m], "F32 pipeline shape diverged");

    // Extract values via the f32 accessor.
    let vals = s.tensor_values_f32(relud).expect("values_f32");
    assert_eq!(vals.len(), n * m, "f32 output length");
    for (i, &v) in vals.iter().enumerate() {
        // After relu, all values >= 0.
        assert!(
            v >= 0.0,
            "relu output[{i}] = {v} should be >= 0 in F32 pipeline"
        );
        assert!(v.is_finite(), "F32 pipeline output[{i}] = {v} non-finite");
    }
});
