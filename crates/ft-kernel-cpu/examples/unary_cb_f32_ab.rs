//! A/B for kgs4.90: f32 compute-bound unary ops were ENTIRELY serial
//! (unary_contiguous_f32 had no parallel path). Now gated at PARALLEL_THRESHOLD
//! (8192). 1-thread vs full-pool of the new kernel IS the exact before/after
//! (old serial == 1-thread). Pure per-element maps → bit-identical (the 465
//! ft-kernel-cpu tests pass).
//!   cargo run -q --release -p ft-kernel-cpu --example unary_cb_f32_ab
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{
    erf_tensor_contiguous_f32, exp_tensor_contiguous_f32, gelu_tensor_contiguous_f32,
    softplus_tensor_contiguous_f32, tanh_tensor_contiguous_f32,
};
use std::time::Instant;

fn bench(name: &str, numel: usize, reps: usize, f: impl Fn(&[f32], &TensorMeta)) {
    let data: Vec<f32> = (0..numel)
        .map(|i| ((i * 2654435761usize) % 4001) as f32 * 0.001 - 2.0)
        .collect();
    let meta = TensorMeta::from_shape(vec![numel], DType::F32, Device::Cpu);
    f(&data, &meta);
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
        std::hint::black_box(erf_tensor_contiguous_f32(d, m).unwrap());
    });
    bench("gelu", numel, reps, |d, m| {
        std::hint::black_box(gelu_tensor_contiguous_f32(d, m).unwrap());
    });
    bench("softplus", numel, reps, |d, m| {
        std::hint::black_box(softplus_tensor_contiguous_f32(d, m).unwrap());
    });
    bench("exp", numel, reps, |d, m| {
        std::hint::black_box(exp_tensor_contiguous_f32(d, m).unwrap());
    });
    bench("tanh", numel, reps, |d, m| {
        std::hint::black_box(tanh_tensor_contiguous_f32(d, m).unwrap());
    });
}

fn main() {
    let nthreads = rayon::current_num_threads();
    {
        let numel = 262_144usize;
        let pool1 = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        println!("numel={numel}  OLD (serial / 1-thread == pre-commit, f32 was fully serial):");
        pool1.install(|| run_all(numel, 40));
        let pooln = rayon::ThreadPoolBuilder::new().build().unwrap();
        println!("numel={numel}  NEW (full pool = {nthreads} threads):");
        pooln.install(|| run_all(numel, 40));
    }
}
