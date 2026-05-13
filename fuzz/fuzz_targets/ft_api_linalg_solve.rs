#![no_main]

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 256;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let n = usize::from(2 + (data[0] % 5)); // 2..=6
    let m = usize::from(1 + (data[1] % 3)); // 1..=3 columns of B
    let body = &data[2..];

    // Build diagonally-dominant A.
    let mut a_data: Vec<f64> = (0..(n * n))
        .map(|i| {
            let raw = body.get(i % body.len().max(1)).copied().unwrap_or(0) as i32;
            (raw - 128) as f64 / 64.0
        })
        .collect();
    for i in 0..n {
        a_data[i * n + i] += n as f64;
    }
    // Build B.
    let b_data: Vec<f64> = (0..(n * m))
        .map(|i| {
            let raw = body.get((n * n + i) % body.len().max(1)).copied().unwrap_or(0) as i32;
            (raw - 128) as f64 / 32.0
        })
        .collect();

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let a = match s.tensor_variable(a_data.clone(), vec![n, n], false) {
        Ok(t) => t,
        Err(_) => return,
    };
    let b = match s.tensor_variable(b_data.clone(), vec![n, m], false) {
        Ok(t) => t,
        Err(_) => return,
    };

    // X = solve(A, B). Then verify A @ X ≈ B.
    let x = match s.tensor_linalg_solve(a, b) {
        Ok(t) => t,
        Err(_) => return,
    };
    let x_vals = s.tensor_values(x).expect("x vals");
    assert_eq!(x_vals.len(), n * m, "solve output length");
    for (i, &v) in x_vals.iter().enumerate() {
        assert!(v.is_finite(), "solve[{i}] = {v} non-finite");
    }

    // A @ X comparison.
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let a2 = s.tensor_variable(a_data.clone(), vec![n, n], false).expect("a2");
    let x2 = s.tensor_variable(x_vals.clone(), vec![n, m], false).expect("x2");
    let prod = s.tensor_matmul(a2, x2).expect("matmul");
    let prod_vals = s.tensor_values(prod).expect("prod vals");

    // Bound: n^3 * EPSILON * |A|max * |X|max for diagonally-dominant A.
    let a_max = a_data.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())).max(1.0);
    let x_max = x_vals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())).max(1.0);
    let bound = (n as f64).powi(3) * f64::EPSILON * a_max * x_max * 64.0 + 1e-9;

    for (i, (got, expected)) in prod_vals.iter().zip(b_data.iter()).enumerate() {
        let diff = (got - expected).abs();
        assert!(
            diff <= bound,
            "(A @ solve(A, B))[{i}] = {got}, expected B[{i}] = {expected}, diff = {diff:e}"
        );
    }
});
