use ft_core::{DType, Device, TensorMeta};
fn fnv(v: &[f32]) -> u64 { let mut a = 0xcbf29ce484222325u64; for x in v { a = (a ^ x.to_bits() as u64).wrapping_mul(0x100000001b3); } a }
fn main() {
    for &(m, k, n) in &[(512usize, 512usize, 512usize), (300usize, 257usize, 259usize)] {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
        let bsq: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.0017).cos()).collect();
        let am = TensorMeta::from_shape(vec![m, k], DType::F32, Device::Cpu);
        let bm = TensorMeta::from_shape(vec![k, n], DType::F32, Device::Cpu);
        let mm = ft_kernel_cpu::matmul_tensor_contiguous_f32(&a, &bsq, &am, &bm).unwrap();
        let wbt: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.0017).cos()).collect();
        let lin = ft_kernel_cpu::linear_tensor_f32(&a, &wbt, None, m, k, n);
        println!("matmul {m}x{k}x{n} fnv={:016x} | linear_bt fnv={:016x}", fnv(&mm), fnv(&lin));
    }
}
