//! A/B for the parallelized Muon Newton-Schulz orthogonalization (frankentorch-optpar).
//! The two matmuls per NS step (X^T@X, X@(1.5I-0.5X^T@X)) were naive serial triple
//! loops — O(rows*cols^2) each, expensive for the large weight matrices Muon targets.
//! Now the output rows fan over rayon (bit-exact, k-order preserved). 1-thread vs
//! all-cores rayon pools, op-only, fresh session/iter, on a 1024x1024 weight matrix.
//!   cargo run -q --release -p ft-optim --example muon_par_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use ft_optim::{Muon, Optimizer};
use rayon::ThreadPool;
use std::time::Instant;

fn time_step(pool: &ThreadPool, rows: usize, cols: usize) -> f64 {
    let n = rows * cols;
    let mut best = f64::INFINITY;
    for _ in 0..6 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable((0..n).map(|i| (i % 97) as f64 * 0.01).collect(), vec![rows, cols], true)
            .unwrap();
        let sq = s.tensor_mul(x, x).unwrap();
        let loss = s.tensor_sum(sq).unwrap();
        let report = s.tensor_backward(loss).unwrap();
        let mut opt = Muon::new(vec![x], 0.02);
        let t = Instant::now();
        pool.install(|| {
            opt.step(&mut s, &report).unwrap();
        });
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() {
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pooln = rayon::ThreadPoolBuilder::new().build().unwrap();
    let nt = pooln.current_num_threads();
    let (rows, cols) = (512usize, 512usize);
    let s1 = time_step(&pool1, rows, cols);
    let sn = time_step(&pooln, rows, cols);
    eprintln!(
        "muon step [{rows}x{cols} weight, Newton-Schulz]: serial {s1:.3} ms / parallel({nt}t) {sn:.3} ms / {:.2}x",
        s1 / sn
    );
}
