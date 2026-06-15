//! Inference (no-grad) before/after for the acosh forward.
//! OLD: serial `.iter().map(f64::acosh)` PLUS the save_for_backward clone the
//! forward closure used to run unconditionally (1x numel f64 memcpy).
//! NEW: production no-grad path — elements mapped over Rayon AND the
//! backward-only clone skipped. Asserts the production output equals the serial
//! reference BIT-FOR-BIT.
//!   cargo run -q --release -p ft-api --example acosh_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_serial_with_save(x: &[f64]) -> Vec<f64> {
    let values: Vec<f64> = x.iter().map(|&xi| xi.acosh()).collect();
    let _saved = std::hint::black_box(x.to_vec());
    values
}

fn main() {
    let nthreads = rayon::current_num_threads();
    let n = 4_000_000usize;
    // acosh domain is x >= 1; spread across [1, ~3.7] so every lane does real work.
    let x: Vec<f64> = (0..n).map(|i| 1.0 + ((i % 271) as f64) * 0.01).collect();
    let want = old_serial_with_save(&x);

    let mut old = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        std::hint::black_box(old_serial_with_save(&x));
        old = old.min(t.elapsed().as_secs_f64() * 1e3);
    }

    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    let new = pn.install(|| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let v = s.tensor_variable(x.clone(), vec![n], false).unwrap();
        let r = s.tensor_acosh(v).unwrap();
        let got = s.tensor_values(r).unwrap();
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.to_bits(), w.to_bits(), "acosh parallel != serial");
        }
        let mut best = f64::INFINITY;
        for _ in 0..15 {
            let v = s.tensor_variable(x.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let r = s.tensor_acosh(v).unwrap();
            std::hint::black_box(r);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    });
    println!(
        "acosh no-grad n={n} (bit-exact OK): OLD serial+save {old:.2}ms  NEW({nthreads}t) {new:.2}ms  =>  {:.2}x",
        old / new
    );
}
