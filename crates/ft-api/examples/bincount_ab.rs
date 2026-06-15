//! A/B for kgs4.101: unweighted bincount binned a serial per-value scatter; now a
//! parallel per-thread-local-bins count (exact integer counts → order-invariant →
//! bit-identical). 1-thread vs full-pool == exact before/after.
//!   cargo run -q --release -p ft-api --example bincount_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn run_once(numel: usize, n_classes: usize, reps: usize) -> f64 {
    let data: Vec<f64> = (0..numel)
        .map(|i| ((i * 2654435761usize) % n_classes) as f64)
        .collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let v = s.tensor_variable(data, vec![numel], false).unwrap();
    let _ = s.tensor_bincount(v, None, 0).unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        std::hint::black_box(s.tensor_bincount(v, None, 0).unwrap());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() {
    let nthreads = rayon::current_num_threads();
    let (numel, n_classes) = (4_000_000usize, 1000usize);
    let p1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let old = p1.install(|| run_once(numel, n_classes, 20));
    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    let new = pn.install(|| run_once(numel, n_classes, 20));
    println!(
        "bincount numel={numel} classes={n_classes}: OLD(1t) {old:.2}ms  NEW({nthreads}t) {new:.2}ms  =>  {:.2}x",
        old / new
    );
}
