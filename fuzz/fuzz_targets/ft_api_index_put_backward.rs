#![no_main]

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 128;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let n = usize::from(2 + (data[0] % 6)); // 2..=7
    let m = usize::from(1 + (data[1] % 4)); // 1..=4 indexed positions
    let body = &data[2..];

    let input: Vec<f64> = (0..n)
        .map(|i| {
            let raw = body.get(i % body.len().max(1)).copied().unwrap_or(0) as i32;
            (raw - 128) as f64 / 25.0
        })
        .collect();
    // Indices into [0, n).
    let indices_data: Vec<f64> = (0..m)
        .map(|i| {
            let raw = body.get((n + i) % body.len().max(1)).copied().unwrap_or(0);
            (usize::from(raw) % n) as f64
        })
        .collect();
    let values_data: Vec<f64> = (0..m).map(|i| (i as f64) + 10.0).collect();

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let input_t = match s.tensor_variable(input.clone(), vec![n], true) {
        Ok(t) => t,
        Err(_) => return,
    };
    let indices_t = match s.tensor_variable(indices_data.clone(), vec![m], false) {
        Ok(t) => t,
        Err(_) => return,
    };
    let values_t = match s.tensor_variable(values_data.clone(), vec![m], true) {
        Ok(t) => t,
        Err(_) => return,
    };

    let result = match s.tensor_index_put(input_t, &[indices_t], values_t, false) {
        Ok(t) => t,
        Err(_) => return,
    };
    let loss = match s.tensor_sum(result) {
        Ok(t) => t,
        Err(_) => return,
    };
    if s.tensor_backward(loss).is_err() {
        return;
    }

    // Input gradient: positions that were OVERWRITTEN by index_put
    // (accumulate=false) should have gradient 0 (the input value
    // doesn't reach the output). Non-overwritten positions should
    // have gradient 1 (since loss = sum and the output equals input
    // there).
    let input_grad_vals: Vec<f64> = match s.tensor_accumulated_gradient(input_t) {
        Ok(Some(v)) => v,
        _ => return,
    };

    let overwritten: std::collections::HashSet<usize> =
        indices_data.iter().map(|&v| v as usize).collect();
    for (i, &g) in input_grad_vals.iter().enumerate() {
        if overwritten.contains(&i) {
            assert!(
                g.abs() < 1e-12,
                "input_grad[{i}] = {g} should be 0 at overwritten position"
            );
        } else {
            assert!(
                (g - 1.0).abs() < 1e-12,
                "input_grad[{i}] = {g} should be 1 at non-overwritten position (loss = sum)"
            );
        }
    }

    // Values gradient: every values position contributes to its
    // overwrite slot in result, which contributes 1 to the sum.
    // So values_grad should be all 1.0 — BUT if multiple indices
    // overwrite the same slot, only the LAST values contribution
    // survives, so some values_grad entries can be 0.
    let values_grad_vals: Vec<f64> = match s.tensor_accumulated_gradient(values_t) {
        Ok(Some(v)) => v,
        _ => return,
    };
    for (i, &g) in values_grad_vals.iter().enumerate() {
        assert!(g.is_finite(), "values_grad[{i}] = {g} non-finite");
        // Values gradient is either 0 (its slot was overwritten by
        // a later index) or 1 (its slot survived).
        assert!(
            g.abs() < 1e-12 || (g - 1.0).abs() < 1e-12,
            "values_grad[{i}] = {g} should be 0 or 1 (got non-binary)"
        );
    }
});
