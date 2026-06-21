use ft_core::{DType, Device, TensorMeta};
fn main() {
    // Deterministic golden of the 2-D-tiled matmul output (square + non-square).
    for &(m, k, n) in &[
        (512usize, 512usize, 512usize),
        (300usize, 257usize, 259usize),
    ] {
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.001).sin()).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.0017).cos()).collect();
        let am = TensorMeta::from_shape(vec![m, k], DType::F64, Device::Cpu);
        let bm = TensorMeta::from_shape(vec![k, n], DType::F64, Device::Cpu);
        let out = ft_kernel_cpu::matmul_tensor_contiguous_f64(&a, &b, &am, &bm).unwrap();
        let mut acc = 0u64;
        for v in &out {
            acc = acc.wrapping_mul(1099511628211).wrapping_add(v.to_bits());
        }
        println!(
            "{m}x{k}x{n} fnv1a={acc:016x} sum={:.12e}",
            out.iter().sum::<f64>()
        );
    }
}
