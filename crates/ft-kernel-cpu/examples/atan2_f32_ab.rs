//! A/B for the f32 binary compute-bound gap: the macro-generated f32 binary
//! kernels (elementwise_contiguous_f32) were ENTIRELY serial, mirroring the
//! pre-kgs4.90 f32 unary gap. atan2 (~one libm atan2 per element) is the
//! compute-bound member; now gated at PARALLEL_THRESHOLD (8192). 1-thread vs
//! full-pool of the new kernel IS the exact before/after (old serial ==
//! 1-thread). Pure per-element map → bit-identical (asserted below).
//!   cargo run -q --release -p ft-kernel-cpu --example atan2_f32_ab
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::atan2_tensor_contiguous_f32;
use std::time::Instant;

fn make(numel: usize) -> (Vec<f32>, Vec<f32>, TensorMeta) {
    let y: Vec<f32> = (0..numel)
        .map(|i| ((i * 2654435761usize) % 4001) as f32 * 0.001 - 2.0)
        .collect();
    let x: Vec<f32> = (0..numel)
        .map(|i| ((i * 40503usize) % 3001) as f32 * 0.001 - 1.5)
        .collect();
    let meta = TensorMeta::from_shape(vec![numel], DType::F32, Device::Cpu);
    (y, x, meta)
}

fn bench(numel: usize, reps: usize) {
    let (y, x, meta) = make(numel);
    let _ = atan2_tensor_contiguous_f32(&y, &x, &meta, &meta).unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        std::hint::black_box(atan2_tensor_contiguous_f32(&y, &x, &meta, &meta).unwrap());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!("  atan2 numel={numel}: {best:.3} ms");
}

fn main() {
    let nthreads = rayon::current_num_threads();
    let numel = 1_048_576usize;

    // Bit-exact proof: 1-thread (old serial) vs full pool, same kernel.
    let (y, x, meta) = make(numel);
    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| atan2_tensor_contiguous_f32(&y, &x, &meta, &meta).unwrap());
    let parall = rayon::ThreadPoolBuilder::new()
        .build()
        .unwrap()
        .install(|| atan2_tensor_contiguous_f32(&y, &x, &meta, &meta).unwrap());
    let bit_exact = serial
        .iter()
        .zip(parall.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    eprintln!("bit-exact (serial == parallel, by bits): {bit_exact}");
    assert!(bit_exact, "parallel atan2 f32 diverged from serial");

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    eprintln!("numel={numel}  OLD (1-thread == pre-commit, f32 binary was fully serial):");
    pool1.install(|| bench(numel, 60));
    let pooln = rayon::ThreadPoolBuilder::new().build().unwrap();
    eprintln!("numel={numel}  NEW (full pool = {nthreads} threads):");
    pooln.install(|| bench(numel, 60));
}
