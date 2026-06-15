//! A/B for kgs4.99: searchsorted (1-D and 2-D batched) ran a serial per-query binary
//! search; now distributes queries across the rayon pool. 1-thread vs full-pool ==
//! exact before/after (old serial == 1-thread). Index order preserved → bit-identical
//! (8 searchsorted tests pass).
//!   cargo run -q --release -p ft-api --example searchsorted_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn run_once(seq_len: usize, n_queries: usize, reps: usize) -> f64 {
    let seq: Vec<f64> = (0..seq_len).map(|i| i as f64 * 2.0).collect(); // sorted bins
    let vals: Vec<f64> = (0..n_queries)
        .map(|i| ((i * 2654435761usize) % (seq_len * 2)) as f64)
        .collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let seqn = s.tensor_variable(seq, vec![seq_len], false).unwrap();
    let valn = s.tensor_variable(vals, vec![n_queries], false).unwrap();
    let _ = s.tensor_searchsorted(seqn, valn, false).unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        std::hint::black_box(s.tensor_searchsorted(seqn, valn, false).unwrap());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() {
    let nthreads = rayon::current_num_threads();
    let (seq_len, n_queries) = (4096usize, 262_144usize);
    let p1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let old = p1.install(|| run_once(seq_len, n_queries, 30));
    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    let new = pn.install(|| run_once(seq_len, n_queries, 30));
    println!(
        "searchsorted 1D bins={seq_len} queries={n_queries}: OLD(1t) {old:.2}ms  NEW({nthreads}t) {new:.2}ms  =>  {:.2}x",
        old / new
    );
}
