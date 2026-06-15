//! A/B for kgs4.101: embedding_bag forward ran a serial per-bag gather+reduce; now
//! parallel over the independent bags. 1-thread vs full-pool == exact before/after.
//! Bit-identical (bags write disjoint output rows; within-bag order preserved).
//!   cargo run -q --release -p ft-api --example embbag_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn run_once(num_bags: usize, bag_sz: usize, num_emb: usize, dim: usize, reps: usize) -> f64 {
    let input_len = num_bags * bag_sz;
    let input: Vec<f64> = (0..input_len)
        .map(|i| ((i * 2654435761usize) % num_emb) as f64)
        .collect();
    let offsets: Vec<f64> = (0..num_bags).map(|b| (b * bag_sz) as f64).collect();
    let weight: Vec<f64> = (0..num_emb * dim).map(|i| (i % 1000) as f64 * 0.001).collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let inp = s.tensor_variable(input, vec![input_len], false).unwrap();
    let w = s.tensor_variable(weight, vec![num_emb, dim], false).unwrap();
    let off = s.tensor_variable(offsets, vec![num_bags], false).unwrap();
    let _ = s.tensor_embedding_bag(inp, w, off, "sum").unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        std::hint::black_box(s.tensor_embedding_bag(inp, w, off, "sum").unwrap());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() {
    let nthreads = rayon::current_num_threads();
    // recsys-style: 8192 bags x 32 items each, weight [50000, 128]
    let (num_bags, bag_sz, num_emb, dim) = (16384usize, 32, 8192, 64);
    let p1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let old = p1.install(|| run_once(num_bags, bag_sz, num_emb, dim, 20));
    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    let new = pn.install(|| run_once(num_bags, bag_sz, num_emb, dim, 20));
    println!(
        "embedding_bag bags={num_bags}x{bag_sz} emb=[{num_emb},{dim}]: OLD(1t) {old:.2}ms  NEW({nthreads}t) {new:.2}ms  =>  {:.2}x",
        old / new
    );
}
