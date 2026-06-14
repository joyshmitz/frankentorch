//! Inference (no-grad) before/after for logaddexp2 forward (always routes through
//! tensor_logaddexp_custom; the finite logaddexp path decomposes through primitives).
//! OLD: serial .iter().zip().map() of stable_logaddexp2_value (log2/exp2 per element)
//! PLUS the two save_for_backward clones the forward closure used to run
//! unconditionally (a_vals.to_vec() + b_vals.to_vec() = 2x numel).
//! NEW: production no-grad path — elements mapped over Rayon AND the backward-only
//! clones skipped. Asserts the production output equals the serial reference BIT-FOR-BIT.
//!   cargo run -q --release -p ft-api --example logaddexp_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

// Mirror of FrankenTorchSession::stable_logaddexp2_value (private) — the exact
// numerically-stable log2(2^a + 2^b) the production op evaluates per element.
fn stable_logaddexp2_value(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    let m = a.max(b);
    if m.is_infinite() {
        return m;
    }
    m + (2.0_f64.powf(a - m) + 2.0_f64.powf(b - m)).log2()
}

fn old_serial_with_saves(a: &[f64], b: &[f64]) -> Vec<f64> {
    let values: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| stable_logaddexp2_value(ai, bi))
        .collect();
    let _x = std::hint::black_box(a.to_vec());
    let _y = std::hint::black_box(b.to_vec());
    values
}

fn main() {
    let nthreads = rayon::current_num_threads();
    let n = 4_000_000usize;
    let a: Vec<f64> = (0..n).map(|i| ((i % 211) as f64) * 0.013 - 1.3).collect();
    let b: Vec<f64> = (0..n).map(|i| ((i % 173) as f64) * 0.011 - 0.9).collect();
    let want = old_serial_with_saves(&a, &b);

    let mut old = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        std::hint::black_box(old_serial_with_saves(&a, &b));
        old = old.min(t.elapsed().as_secs_f64() * 1e3);
    }

    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    let new = pn.install(|| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let va = s.tensor_variable(a.clone(), vec![n], false).unwrap();
        let vb = s.tensor_variable(b.clone(), vec![n], false).unwrap();
        let r = s.tensor_logaddexp2(va, vb).unwrap();
        let got = s.tensor_values(r).unwrap();
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.to_bits(), w.to_bits(), "logaddexp2 parallel != serial");
        }
        let mut best = f64::INFINITY;
        for _ in 0..15 {
            let va = s.tensor_variable(a.clone(), vec![n], false).unwrap();
            let vb = s.tensor_variable(b.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let r = s.tensor_logaddexp2(va, vb).unwrap();
            std::hint::black_box(r);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    });
    println!(
        "logaddexp2 no-grad n={n} (bit-exact OK): OLD serial+saves {old:.2}ms  NEW({nthreads}t) {new:.2}ms  =>  {:.2}x",
        old / new
    );
}
