//! Before/after for the shared in-place unary helper apply_tensor_unary_in_place,
//! which backs 73 in-place ops (exp_/ln_/tanh_/sin_/sqrt_/erfinv_/...).
//! OLD: serial `target_vals.into_iter().map(transform).collect()`.
//! NEW: the per-element transform is fanned over Rayon for tensors >=
//! PARALLEL_ELEMENTWISE_MIN. Order-preserving => bit-for-bit identical.
//! Exercised here via tensor_exp_ (a heavy transcendental representative).
//!   cargo run -q --release -p ft-api --example inplace_unary_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_serial(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| v.exp()).collect()
}

fn main() {
    let nthreads = rayon::current_num_threads();
    let n = 4_000_000usize;
    let x: Vec<f64> = (0..n).map(|i| ((i % 233) as f64) * 0.01 - 1.1).collect();
    let want = old_serial(&x);

    let mut old = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        std::hint::black_box(old_serial(&x));
        old = old.min(t.elapsed().as_secs_f64() * 1e3);
    }

    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    let new = pn.install(|| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Correctness: in-place exp_ output must be bit-for-bit == serial exp.
        let v = s.tensor_variable(x.clone(), vec![n], false).unwrap();
        s.tensor_exp_(v).unwrap();
        let got = s.tensor_values(v).unwrap();
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.to_bits(), w.to_bits(), "in-place exp_ parallel != serial");
        }
        let mut best = f64::INFINITY;
        for _ in 0..15 {
            let v = s.tensor_variable(x.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            s.tensor_exp_(v).unwrap();
            std::hint::black_box(v);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    });
    println!(
        "in-place exp_ n={n} (bit-exact OK): OLD serial {old:.2}ms  NEW({nthreads}t) {new:.2}ms  =>  {:.2}x",
        old / new
    );
}
