//! A/B for the f32 softmax gap: softmax_dim_tensor_contiguous_f32 was fully
//! serial while its f64 twin (and f32 log_softmax) already parallelise over
//! independent rows. exp(x-max) per element + pairwise sum is compute-bound;
//! rows are independent so par == serial bit-for-bit. 1-thread vs full pool of
//! the new kernel IS the exact before/after (old serial == 1-thread).
//!   cargo run -q --release -p ft-kernel-cpu --example softmax_f32_ab
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::softmax_dim_tensor_contiguous_f32;
use std::time::Instant;

fn make(rows: usize, cols: usize) -> (Vec<f32>, TensorMeta) {
    let n = rows * cols;
    let data: Vec<f32> = (0..n)
        .map(|i| ((i * 2654435761usize) % 9001) as f32 * 0.001 - 4.0)
        .collect();
    let meta = TensorMeta::from_shape(vec![rows, cols], DType::F32, Device::Cpu);
    (data, meta)
}

fn bench(rows: usize, cols: usize, reps: usize) {
    let (data, meta) = make(rows, cols);
    let _ = softmax_dim_tensor_contiguous_f32(&data, &meta, 1).unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        std::hint::black_box(softmax_dim_tensor_contiguous_f32(&data, &meta, 1).unwrap());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!("  softmax [{rows},{cols}] dim=1: {best:.3} ms");
}

fn main() {
    let nthreads = rayon::current_num_threads();
    // Classifier/attention shape: softmax over last dim, many rows.
    let (rows, cols) = (16_384usize, 256usize); // numel 4.19M, > gates

    // Bit-exact proof: 1-thread (old serial) vs full pool, same kernel.
    let (data, meta) = make(rows, cols);
    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| softmax_dim_tensor_contiguous_f32(&data, &meta, 1).unwrap());
    let parall = rayon::ThreadPoolBuilder::new()
        .build()
        .unwrap()
        .install(|| softmax_dim_tensor_contiguous_f32(&data, &meta, 1).unwrap());
    let bit_exact = serial
        .iter()
        .zip(parall.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    eprintln!("bit-exact (serial == parallel, by bits): {bit_exact}");
    assert!(bit_exact, "parallel softmax f32 diverged from serial");

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    eprintln!("OLD (1-thread == pre-commit, f32 softmax was fully serial):");
    pool1.install(|| bench(rows, cols, 40));
    let pooln = rayon::ThreadPoolBuilder::new().build().unwrap();
    eprintln!("NEW (full pool = {nthreads} threads):");
    pooln.install(|| bench(rows, cols, 40));
}
