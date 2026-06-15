use ft_core::{DType, Device, TensorMeta};
fn fnv(v: &[f64]) -> u64 {
    let mut a = 0xcbf29ce484222325u64;
    for x in v {
        a = (a ^ x.to_bits()).wrapping_mul(0x100000001b3);
    }
    a
}
fn main() {
    for &(m, k, n) in &[
        (256usize, 64usize, 16384usize),
        (64usize, 256usize, 8192usize),
    ] {
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.001).sin()).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.0017).cos()).collect();
        let am = TensorMeta::from_shape(vec![m, k], DType::F64, Device::Cpu);
        let bm = TensorMeta::from_shape(vec![k, n], DType::F64, Device::Cpu);
        let o = ft_kernel_cpu::matmul_tensor_contiguous_f64(&a, &b, &am, &bm).unwrap();
        println!("wide {m}x{k}x{n} fnv={:016x}", fnv(&o));
    }
}
