//! A/B for parallelizing elementwise_strided_f64 (the non-contiguous same-shape
//! binary kernel, reached by e.g. `a.t().atan2(b.t())` without a copy). It was a
//! serial coords-increment sweep; now a per-flat-index unravel + into_par_iter at
//! PARALLEL_THRESHOLD. Because the kernel only parallelizes above the threshold,
//! ThreadPoolBuilder(1) reproduces the OLD serial path EXACTLY and the full pool
//! is the NEW parallel path — so 1t-vs-Nt on the SAME kernel is the true pre/post.
//! atan2 is a transcendental (compute-bound). Result asserted bit-for-bit (1t==Nt).
//!   cargo run -q --release -p ft-kernel-cpu --example strided_binary_ab
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{atan2_tensor_contiguous_f64, erf_tensor_contiguous_f64};
use std::time::Instant;

fn transposed_meta(rows: usize, cols: usize) -> TensorMeta {
    // Logical [cols, rows] transposed view of a row-major [rows, cols] buffer:
    // shape [cols, rows], strides [1, cols] => non-contiguous, hits the strided path.
    TensorMeta::from_shape_and_strides(vec![cols, rows], vec![1, cols], 0, DType::F64, Device::Cpu)
        .unwrap()
}

fn bench(label: &str, rows: usize, cols: usize) {
    let n = rows * cols;
    let lhs: Vec<f64> = (0..n).map(|i| ((i % 617) as f64) * 0.01 - 3.0).collect();
    let rhs: Vec<f64> = (0..n).map(|i| ((i % 953) as f64) * 0.01 + 0.5).collect();
    let meta = transposed_meta(rows, cols);

    let run = || atan2_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).unwrap();

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pooln = rayon::ThreadPoolBuilder::new().build().unwrap();
    let nthreads = pooln.current_num_threads();

    let serial = pool1.install(run);
    let parallel = pooln.install(run);
    assert_eq!(serial.len(), parallel.len());
    for (a, b) in serial.iter().zip(parallel.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "parallel strided != serial");
    }

    let mut old = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        std::hint::black_box(pool1.install(run));
        old = old.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut new = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        std::hint::black_box(pooln.install(run));
        new = new.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "{label} [{cols}x{rows}={n}] (1t vs {nthreads}t, bit-exact OK): serial {old:.3}ms  par {new:.3}ms  =>  {:.2}x",
        old / new
    );
}

fn bench_unary(label: &str, rows: usize, cols: usize) {
    let n = rows * cols;
    let x: Vec<f64> = (0..n).map(|i| ((i % 617) as f64) * 0.006 - 1.8).collect();
    let meta = transposed_meta(rows, cols);
    let run = || erf_tensor_contiguous_f64(&x, &meta).unwrap();

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pooln = rayon::ThreadPoolBuilder::new().build().unwrap();
    let nthreads = pooln.current_num_threads();
    let serial = pool1.install(run);
    let parallel = pooln.install(run);
    for (a, b) in serial.iter().zip(parallel.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "parallel unary strided != serial");
    }
    let mut old = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        std::hint::black_box(pool1.install(run));
        old = old.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut new = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        std::hint::black_box(pooln.install(run));
        new = new.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "{label} [{cols}x{rows}={n}] (1t vs {nthreads}t, bit-exact OK): serial {old:.3}ms  par {new:.3}ms  =>  {:.2}x",
        old / new
    );
}

fn main() {
    bench("ANCHOR-small", 32, 32);
    bench("strided-atan2", 1024, 1024);
    bench("strided-atan2", 2048, 2048);
    bench_unary("strided-erf", 1024, 1024);
    bench_unary("strided-erf", 2048, 2048);
}
