// Wide-matmul (n >> m, col-parallel path) throughput via the public entry.
use ft_core::{DType, Device, TensorMeta};
use std::time::Instant;
fn main() {
    println!("threads = {}", rayon::current_num_threads());
    for &(m, k, n) in &[(256usize, 64, 16384), (128, 1024, 4096), (64, 256, 8192)] {
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.001).sin()).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.0017).cos()).collect();
        let am = TensorMeta::from_shape(vec![m, k], DType::F64, Device::Cpu);
        let bm = TensorMeta::from_shape(vec![k, n], DType::F64, Device::Cpu);
        for _ in 0..3 {
            let _ = ft_kernel_cpu::matmul_tensor_contiguous_f64(&a, &b, &am, &bm).unwrap();
        }
        let mut best = f64::MAX;
        for _ in 0..6 {
            let s = Instant::now();
            for _ in 0..10 {
                let _ = ft_kernel_cpu::matmul_tensor_contiguous_f64(&a, &b, &am, &bm).unwrap();
            }
            best = best.min(s.elapsed().as_secs_f64() / 10.0 * 1000.0);
        }
        println!(
            "wide m{m} k{k} n{n}: {best:.3} ms  {:.0} GFLOP/s",
            2.0 * m as f64 * k as f64 * n as f64 / 1e9 / (best / 1e3)
        );
    }
}
