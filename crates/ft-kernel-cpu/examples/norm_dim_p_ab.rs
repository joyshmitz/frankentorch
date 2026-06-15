//! A/B for the general-p norm_dim gap: norm_dim_tensor_contiguous_f64/f32 ran
//! the general fractional-p branch (one powf/element) fully serial while the
//! cheap p=1/2/inf branches stay serial (bandwidth-bound). The powf branch is
//! compute-bound → parallel over independent output rows, bit-exact. 1-thread
//! vs full pool of the new kernel IS the exact before/after (old serial ==
//! 1-thread, since the parallel path only triggers above the work gate).
//!   cargo run -q --release -p ft-kernel-cpu --example norm_dim_p_ab
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::norm_dim_tensor_contiguous_f64;
use std::time::Instant;

fn make(shape: &[usize]) -> (Vec<f64>, TensorMeta) {
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n).map(|i| ((i % 9973) as f64) * 0.0007 - 3.0).collect();
    let meta = TensorMeta::from_shape(shape.to_vec(), DType::F64, Device::Cpu);
    (data, meta)
}

fn bench(shape: &[usize], p: f64, reps: usize) {
    let (data, meta) = make(shape);
    let _ = norm_dim_tensor_contiguous_f64(&data, &meta, p, 1).unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        std::hint::black_box(norm_dim_tensor_contiguous_f64(&data, &meta, p, 1).unwrap());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!("  norm p={p} {shape:?} dim=1: {best:.3} ms");
}

fn main() {
    let nthreads = rayon::current_num_threads();
    // out_numel = 4096 rows, reduce_size = 1024 → big compute-bound powf sweep.
    let shape = vec![4096usize, 1024, 1];
    let p = 3.0;

    let (data, meta) = make(&shape);
    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| norm_dim_tensor_contiguous_f64(&data, &meta, p, 1).unwrap());
    let parall = rayon::ThreadPoolBuilder::new()
        .build()
        .unwrap()
        .install(|| norm_dim_tensor_contiguous_f64(&data, &meta, p, 1).unwrap());
    let bit_exact = serial
        .iter()
        .zip(parall.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    eprintln!("bit-exact (serial == parallel, by bits): {bit_exact}");
    assert!(bit_exact, "parallel norm_dim general-p diverged from serial");

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    eprintln!("OLD (1-thread == pre-commit, general-p was serial):");
    pool1.install(|| bench(&shape, p, 40));
    let pooln = rayon::ThreadPoolBuilder::new().build().unwrap();
    eprintln!("NEW (full pool = {nthreads} threads):");
    pooln.install(|| bench(&shape, p, 40));
}
