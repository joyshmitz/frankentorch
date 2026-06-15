//! A/B for kgs4.95: reduce_sum_for_broadcast (the backward of broadcasting — hit on
//! every training step through a broadcast add/mul) was a serial row-major scatter-add.
//! Reparallelized over OUTPUTS, each summing its contributors in the same increasing-
//! flat-index order (bit-exact; proven by *_parallel_matches_serial_scatter_bit_exact).
//! 1-thread vs full-pool == exact before/after (old serial == 1-thread, since the
//! per-output index decode dominates the single read → compute-bound).
//!   cargo run -q --release -p ft-kernel-cpu --example reduce_bcast_ab
use ft_kernel_cpu::reduce_sum_for_broadcast;
use std::time::Instant;

fn bench(name: &str, reps: usize, f: impl Fn()) {
    f();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("  {name:<22}: {best:.3} ms");
}

fn main() {
    let nthreads = rayon::current_num_threads();
    // [4096,1024] -> [1,1024]: bias-grad style reduction over the leading dim, 4.2M in.
    let exp = [4096usize, 1024];
    let orig = [1usize, 1024];
    let numel: usize = exp.iter().product();
    let grad: Vec<f64> = (0..numel)
        .map(|i| (i % 1000) as f64 * 0.001 - 0.5)
        .collect();
    let run = || {
        std::hint::black_box(reduce_sum_for_broadcast(&grad, &exp, &orig).unwrap());
    };
    let p1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    println!("reduce_sum_for_broadcast [4096,1024]->[1,1024]:");
    p1.install(|| bench("OLD (1 thread)", 40, run));
    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    pn.install(|| bench(&format!("NEW ({nthreads} threads)"), 40, run));
}
