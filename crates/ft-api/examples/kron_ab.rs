//! A/B for kgs4.102: kron (no-grad) ran a serial 4-deep loop; now a parallel
//! per-output-index-unravel map. The serial reference (old incremental loop) is timed
//! directly for an honest before/after. Bit-identical (indexed map preserves order).
//!   cargo run -q --release -p ft-api --example kron_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_serial(a: &[f64], b: &[f64], m: usize, n: usize, p: usize, q: usize) -> Vec<f64> {
    let mut result = vec![0.0f64; m * p * n * q];
    for i in 0..m {
        for j in 0..n {
            let av = a[i * n + j];
            for k in 0..p {
                for l in 0..q {
                    result[(i * p + k) * (n * q) + (j * q + l)] = av * b[k * q + l];
                }
            }
        }
    }
    result
}

fn main() {
    let nthreads = rayon::current_num_threads();
    let (m, n, p, q) = (128usize, 128, 32, 32); // out = [4096, 4096] = 16.7M
    let a: Vec<f64> = (0..m * n).map(|i| (i % 1000) as f64 * 0.001).collect();
    let b: Vec<f64> = (0..p * q).map(|i| (i % 997) as f64 * 0.001).collect();

    // OLD: serial 4-deep loop (the pre-commit algorithm), timed directly.
    let mut old = f64::INFINITY;
    for _ in 0..10 {
        let t = Instant::now();
        std::hint::black_box(old_serial(&a, &b, m, n, p, q));
        old = old.min(t.elapsed().as_secs_f64() * 1e3);
    }

    // NEW: production kron (parallel) at full pool.
    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    let new = pn.install(|| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let an = s.tensor_variable(a.clone(), vec![m, n], false).unwrap();
        let bn = s.tensor_variable(b.clone(), vec![p, q], false).unwrap();
        let _ = s.kron(an, bn).unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..10 {
            let t = Instant::now();
            std::hint::black_box(s.kron(an, bn).unwrap());
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    });
    println!(
        "kron [{m},{n}]⊗[{p},{q}] -> [4096,4096]: OLD serial {old:.2}ms  NEW({nthreads}t) {new:.2}ms  =>  {:.2}x",
        old / new
    );
}
