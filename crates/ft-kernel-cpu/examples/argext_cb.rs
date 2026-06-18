use ft_core::{DType, Device, TensorMeta};
use rayon::prelude::*;
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
// OLD per-lane strided argmax (what shipped last session) for d0.
fn old_argmax_d0(d: &[f64], inner: usize, red: usize) -> Vec<f64> {
    let mut out = vec![0.0; inner];
    out.par_iter_mut().enumerate().for_each(|(c, o)| {
        let mut bi = 0usize;
        let mut bv = f64::NEG_INFINITY;
        for r in 0..red {
            let v = d[r * inner + c];
            if v.is_nan() {
                bi = r;
                break;
            } else if v > bv {
                bv = v;
                bi = r;
            }
        }
        *o = bi as f64;
    });
    out
}
fn main() {
    println!("threads={}", rayon::current_num_threads());
    let n = 2048;
    let d: Vec<f64> = (0..n * n)
        .map(|i| ((i * 2654435761usize) % 10007) as f64 * 0.001)
        .collect();
    let m = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
    let old = t(|| {
        let _ = old_argmax_d0(&d, n, n);
    });
    let new = t(|| {
        let _ = ft_kernel_cpu::argmax_dim_tensor_contiguous_f64(&d, &m, 0).unwrap();
    });
    println!(
        "d0: old(strided) {old:.3}ms | new(cache-blocked) {new:.3}ms | {:.2}x",
        old / new
    );
}
