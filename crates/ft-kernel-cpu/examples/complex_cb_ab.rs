//! A/B for kgs4.91: compute-bound complex elementwise ops (abs/angle/mul/div) were
//! fully serial. Now gated at PARALLEL_THRESHOLD (8192). 1-thread vs full-pool of the
//! new kernel IS the exact before/after (old serial == 1-thread). Pure per-element
//! maps → bit-identical (the complex unit tests pass).
//!   cargo run -q --release -p ft-kernel-cpu --example complex_cb_ab
use ft_core::{Complex128, DType, Device, TensorMeta};
use ft_kernel_cpu::{
    complex_abs_contiguous, complex_angle_contiguous, complex_div_contiguous,
    complex_mul_contiguous,
};
use std::time::Instant;

fn data(numel: usize) -> Vec<Complex128> {
    (0..numel)
        .map(|i| {
            let r = ((i * 2654435761usize) % 4001) as f64 * 0.001 - 2.0;
            let im = ((i * 40503usize) % 4001) as f64 * 0.001 - 2.0;
            Complex128::new(r, im)
        })
        .collect()
}

fn bench1(
    name: &str,
    d: &[Complex128],
    m: &TensorMeta,
    reps: usize,
    f: impl Fn(&[Complex128], &TensorMeta),
) {
    f(d, m);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f(d, m);
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("  {name:<8}: {best:.3} ms");
}

fn run_all(numel: usize, reps: usize) {
    let d = data(numel);
    let m = TensorMeta::from_shape(vec![numel], DType::Complex128, Device::Cpu);
    bench1("abs", &d, &m, reps, |d, m| {
        std::hint::black_box(complex_abs_contiguous(d, m).unwrap());
    });
    bench1("angle", &d, &m, reps, |d, m| {
        std::hint::black_box(complex_angle_contiguous(d, m).unwrap());
    });
    bench1("mul", &d, &m, reps, |d, m| {
        std::hint::black_box(complex_mul_contiguous(d, d, m, m).unwrap());
    });
    bench1("div", &d, &m, reps, |d, m| {
        std::hint::black_box(complex_div_contiguous(d, d, m, m).unwrap());
    });
}

fn main() {
    let nthreads = rayon::current_num_threads();
    let numel = 262_144usize;
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    println!("numel={numel}  OLD (serial / 1-thread == pre-commit):");
    pool1.install(|| run_all(numel, 40));
    let pooln = rayon::ThreadPoolBuilder::new().build().unwrap();
    println!("numel={numel}  NEW (full pool = {nthreads} threads):");
    pooln.install(|| run_all(numel, 40));
}
