//! A/B for kgs4.89: compute-bound unary ops (erf/gelu/silu/softplus) parallelized
//! at the low PARALLEL_THRESHOLD (8192). For numel in [8192, 524288) the OLD code
//! ran these SERIALLY (the cheap-op SCALAR_UNARY_PARALLEL_THRESHOLD = 524288), so
//! a 1-thread vs full-pool run of the new kernel IS the exact before/after (old
//! serial == 1-thread). Pure per-element maps → parallel is bit-identical (proven
//! by the existing *_matches_elementwise_bit_exact unit tests).
//!   cargo run -q --release -p ft-kernel-cpu --example unary_cb_ab
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{
    erf_tensor_contiguous_f64, gelu_tensor_contiguous_f64, silu_tensor_contiguous_f64,
    softplus_tensor_contiguous_f64,
};
use std::time::Instant;

fn bench(name: &str, numel: usize, reps: usize, f: impl Fn(&[f64], &TensorMeta)) {
    let data: Vec<f64> = (0..numel)
        .map(|i| ((i * 2654435761usize) % 4001) as f64 * 0.001 - 2.0)
        .collect();
    let meta = TensorMeta::from_shape(vec![numel], DType::F64, Device::Cpu);
    f(&data, &meta); // warm
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f(&data, &meta);
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("  {name:<10} numel={numel}: {best:.3} ms");
}

fn run_all(numel: usize, reps: usize) {
    bench("erf", numel, reps, |d, m| {
        std::hint::black_box(erf_tensor_contiguous_f64(d, m).unwrap());
    });
    bench("gelu", numel, reps, |d, m| {
        std::hint::black_box(gelu_tensor_contiguous_f64(d, m).unwrap());
    });
    bench("silu", numel, reps, |d, m| {
        std::hint::black_box(silu_tensor_contiguous_f64(d, m).unwrap());
    });
    bench("softplus", numel, reps, |d, m| {
        std::hint::black_box(softplus_tensor_contiguous_f64(d, m).unwrap());
    });
}

fn main() {
    let nthreads = rayon::current_num_threads();
    for &numel in &[65_536usize, 262_144] {
        let pool1 = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        println!("numel={numel}  OLD (serial / 1-thread, == pre-commit below 524288):");
        pool1.install(|| run_all(numel, 30));
        let pooln = rayon::ThreadPoolBuilder::new().build().unwrap();
        println!("numel={numel}  NEW (full pool = {nthreads} threads):");
        pooln.install(|| run_all(numel, 30));
    }
}
