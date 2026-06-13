// GEMM throughput aid (public API). Square f64 matmul GFLOP/s across sizes.
use ft_core::{DType, Device, TensorMeta};
use std::time::Instant;
fn main() {
    println!("threads = {}", rayon::current_num_threads());
    for &n in &[512usize, 1024, 2048] {
        let a: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.001).sin()).collect();
        let b: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.002).cos()).collect();
        let am = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let bm = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        for _ in 0..3 {
            let _ = ft_kernel_cpu::matmul_tensor_contiguous_f64(&a, &b, &am, &bm).unwrap();
        }
        let mut best = f64::MAX;
        for _ in 0..6 {
            let t = Instant::now();
            for _ in 0..10 {
                let _ = ft_kernel_cpu::matmul_tensor_contiguous_f64(&a, &b, &am, &bm).unwrap();
            }
            best = best.min(t.elapsed().as_secs_f64() / 10.0 * 1000.0);
        }
        println!(
            "n={n:>4}: {best:>8.3} ms  {:>7.1} GFLOP/s",
            2.0 * (n as f64).powi(3) / 1e9 / (best / 1000.0)
        );
    }
}
