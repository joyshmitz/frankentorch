//! Golden-fixture generator for the Cholesky column-update perf investigation
//! (`frankentorch-kgs4.1`). Prints the bit-for-bit serial factor of a fixed,
//! well-conditioned SPD matrix so the canonical reference is pinned even though
//! the parallel lever was rejected (memory-bandwidth bound, < Score 2.0).
//!
//!   rch exec -- cargo run -p ft-kernel-cpu --example cholesky_golden

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::cholesky_contiguous_f64;

fn main() {
    let n = 6usize;
    // SPD: A = B^T B + n*I (well-conditioned, positive definite).
    let mut b = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            b[i * n + j] = ((i * 31 + j * 17) % 97) as f64 * 0.013 - 0.5;
        }
    }
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0_f64;
            for k in 0..n {
                s += b[k * n + i] * b[k * n + j];
            }
            a[i * n + j] = s;
        }
        a[i * n + i] += n as f64;
    }
    let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
    let r = cholesky_contiguous_f64(&a, &meta, false).unwrap();

    println!("frankentorch-kgs4.1 cholesky_serial_golden");
    println!("n={}", r.n);
    println!("factor_bits:");
    for (idx, v) in r.factor.iter().enumerate() {
        println!("{idx}: {:#018x}", v.to_bits());
    }
}
