//! Premise check for native-f32 det wiring: is f32 LU >= 2x f64 LU? (det's O(n^3) cost)
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{lu_factor_contiguous_f32, lu_factor_contiguous_f64};
fn main() {
    for n in [256usize, 512, 1024] {
        let a64: Vec<f64> = (0..n * n)
            .map(|i| {
                ((i * 1103515245 + 12345) % 1000) as f64 * 0.001
                    + if i % (n + 1) == 0 { n as f64 } else { 0.0 }
            })
            .collect();
        let a32: Vec<f32> = a64.iter().map(|&x| x as f32).collect();
        let m64 = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let m32 = TensorMeta::from_shape(vec![n, n], DType::F32, Device::Cpu);
        let mut b64 = f64::INFINITY;
        for _ in 0..10 {
            let t = std::time::Instant::now();
            std::hint::black_box(lu_factor_contiguous_f64(&a64, &m64).unwrap());
            b64 = b64.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let mut b32 = f64::INFINITY;
        for _ in 0..10 {
            let t = std::time::Instant::now();
            std::hint::black_box(lu_factor_contiguous_f32(&a32, &m32).unwrap());
            b32 = b32.min(t.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "LU n={n}: f64 {b64:.3}ms  f32 {b32:.3}ms  =>  {:.2}x",
            b64 / b32
        );
    }
}
