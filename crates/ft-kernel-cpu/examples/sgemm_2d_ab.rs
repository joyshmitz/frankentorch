// f32 throughput aid via the public entries: linear_tensor_f32 (x@W^T = sgemm_bt)
// and matmul_tensor_contiguous_f32 (sgemm). Both now use K-gated 2-D tiling.
use ft_core::{DType, Device, TensorMeta};
use std::time::Instant;
fn timeit<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..3 {
        f();
    }
    let mut best = f64::MAX;
    for _ in 0..6 {
        let s = Instant::now();
        for _ in 0..10 {
            f();
        }
        best = best.min(s.elapsed().as_secs_f64() / 10.0 * 1000.0);
    }
    best
}
fn main() {
    println!("threads = {}", rayon::current_num_threads());
    for &(m, k, n) in &[
        (512usize, 1024, 1024),
        (1024, 1024, 1024),
        (4096, 256, 1024),
        (2048, 2048, 2048),
    ] {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.0017).cos()).collect();
        let g = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
        let lin = timeit(|| {
            let _ = ft_kernel_cpu::linear_tensor_f32(&a, &w, None, m, k, n);
        });
        let am = TensorMeta::from_shape(vec![m, k], DType::F32, Device::Cpu);
        let bm = TensorMeta::from_shape(vec![k, n], DType::F32, Device::Cpu);
        let bsq: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.0017).cos()).collect();
        let mm = timeit(|| {
            let _ = ft_kernel_cpu::matmul_tensor_contiguous_f32(&a, &bsq, &am, &bm).unwrap();
        });
        println!(
            "m{m:>4} k{k:>4} n{n:>4}: linear(bt) {:>5.0}GF {lin:.3}ms | matmul {:>5.0}GF {mm:.3}ms",
            g / (lin / 1e3),
            g / (mm / 1e3)
        );
    }
}
