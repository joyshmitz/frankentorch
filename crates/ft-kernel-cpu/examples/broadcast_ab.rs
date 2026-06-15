//! A/B for kgs4.93: broadcast binary ops (add/sub/mul/div via elementwise_broadcast_f64)
//! were serial. Each element unravels its flat index into two source indices
//! (per-dim div/mod over both operands) — that index math dominates, so the loop is
//! compute-bound and parallelizes. 1-thread vs full-pool == exact before/after (old
//! serial == 1-thread). Pure indexed map → bit-identical (broadcast unit tests pass).
//!   cargo run -q --release -p ft-kernel-cpu --example broadcast_ab
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{add_tensor_broadcast_f64, mul_tensor_broadcast_f64};
use std::time::Instant;

fn bench(name: &str, reps: usize, f: impl Fn()) {
    f();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("  {name:<14}: {best:.3} ms");
}

fn run_all(reps: usize) {
    // [rows, cols] (+) [cols]  — classic bias broadcast over rows. out = rows*cols.
    let rows = 2048usize;
    let cols = 2048usize;
    let lhs: Vec<f64> = (0..rows * cols)
        .map(|i| (i % 1000) as f64 * 0.001)
        .collect();
    let rhs: Vec<f64> = (0..cols).map(|i| (i % 997) as f64 * 0.001).collect();
    let lm = TensorMeta::from_shape(vec![rows, cols], DType::F64, Device::Cpu);
    let rm = TensorMeta::from_shape(vec![cols], DType::F64, Device::Cpu);
    bench("bcast_add 2k", reps, || {
        std::hint::black_box(add_tensor_broadcast_f64(&lhs, &rhs, &lm, &rm).unwrap());
    });
    bench("bcast_mul 2k", reps, || {
        std::hint::black_box(mul_tensor_broadcast_f64(&lhs, &rhs, &lm, &rm).unwrap());
    });
}

fn main() {
    let nthreads = rayon::current_num_threads();
    let p1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    println!("OLD (serial / 1-thread == pre-commit):");
    p1.install(|| run_all(40));
    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    println!("NEW (full pool = {nthreads} threads):");
    pn.install(|| run_all(40));
}
