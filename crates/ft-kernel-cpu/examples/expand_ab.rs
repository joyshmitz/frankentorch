//! A/B for kgs4.94: expand (broadcast materialization, the path behind f32
//! broadcasting + explicit .expand()) was a serial coords-sweep gather. Each output
//! element unravels its flat index (per-dim div/mod) → compute-bound, parallelizes.
//! 1-thread vs full-pool == exact before/after (old serial == 1-thread). Indexed map
//! → bit-identical (expand unit tests pass).
//!   cargo run -q --release -p ft-kernel-cpu --example expand_ab
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{expand_tensor_contiguous_f32, expand_tensor_contiguous_f64};
use std::time::Instant;

fn bench(name: &str, reps: usize, f: impl Fn()) {
    f();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("  {name:<16}: {best:.3} ms");
}

fn run_all(reps: usize) {
    let m = 2048usize;
    let n = 2048usize; // [1, m] -> [n, m], out = n*m = 4.2M
    let row64: Vec<f64> = (0..m).map(|i| (i % 1000) as f64 * 0.001).collect();
    let row32: Vec<f32> = (0..m).map(|i| (i % 1000) as f32 * 0.001).collect();
    let im = TensorMeta::from_shape(vec![1, m], DType::F64, Device::Cpu);
    let im32 = TensorMeta::from_shape(vec![1, m], DType::F32, Device::Cpu);
    bench("expand_f64 2k", reps, || {
        std::hint::black_box(expand_tensor_contiguous_f64(&row64, &im, &[n, m]).unwrap());
    });
    bench("expand_f32 2k", reps, || {
        std::hint::black_box(expand_tensor_contiguous_f32(&row32, &im32, &[n, m]).unwrap());
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
