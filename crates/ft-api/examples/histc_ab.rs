//! A/B for kgs4.100: histc binned a serial per-value scatter; now a parallel
//! per-thread-local-bins histogram (counts are exact integers → order-invariant →
//! bit-identical). 1-thread vs full-pool == exact before/after.
//!   cargo run -q --release -p ft-api --example histc_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn run_once(numel: usize, bins: usize, reps: usize) -> f64 {
    let data: Vec<f64> = (0..numel)
        .map(|i| ((i * 2654435761usize) % 10_000) as f64 * 0.01)
        .collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let v = s.tensor_variable(data, vec![numel], false).unwrap();
    let _ = s.tensor_histc(v, bins, 0.0, 100.0).unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        std::hint::black_box(s.tensor_histc(v, bins, 0.0, 100.0).unwrap());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() {
    let nthreads = rayon::current_num_threads();
    let (numel, bins) = (4_000_000usize, 256usize);
    let p1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let old = p1.install(|| run_once(numel, bins, 20));
    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    let new = pn.install(|| run_once(numel, bins, 20));
    println!(
        "histc numel={numel} bins={bins}: OLD(1t) {old:.2}ms  NEW({nthreads}t) {new:.2}ms  =>  {:.2}x",
        old / new
    );
}
