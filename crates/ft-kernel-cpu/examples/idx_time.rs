use ft_core::{DType, Device, TensorMeta};
use std::time::Instant;
fn t<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..3 {
        f();
    }
    let mut b = f64::MAX;
    for _ in 0..8 {
        let s = Instant::now();
        for _ in 0..10 {
            f();
        }
        b = b.min(s.elapsed().as_secs_f64() / 10.0 * 1000.0);
    }
    b
}
fn serial_gather1(d: &[f64], gidx: &[f64], n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n * n);
    for outer in 0..n {
        for r in 0..n {
            out.push(d[outer * n + gidx[outer * n + r] as usize]);
        }
    }
    out
}
fn main() {
    println!("threads={}", rayon::current_num_threads());
    let n = 2048;
    let d: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.001).collect();
    let m = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
    let gidx: Vec<f64> = (0..n * n).map(|i| ((i * 13) % n) as f64).collect();
    let im = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
    let old = t(|| {
        let _ = serial_gather1(&d, &gidx, n);
    });
    let new = t(|| {
        let _ = ft_kernel_cpu::gather_tensor_contiguous_f64(&d, &m, 1, &gidx, &im).unwrap();
    });
    println!(
        "gather1: serial {old:.3}ms | parallel {new:.3}ms | {:.2}x",
        old / new
    );
}
