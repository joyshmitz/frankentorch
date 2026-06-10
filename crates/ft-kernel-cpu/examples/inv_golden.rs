//! Deterministic matrix-inverse golden (frankentorch inv gate-threshold perf).
//!
//! Emits the bit pattern of every entry of A^-1 for a few well-conditioned
//! square sizes. Piping through `sha256sum` gives a before/after isomorphism
//! proof when changing the lu_solve column-parallel gate (the serial and
//! column-parallel substitution paths must produce bit-identical inverses).
//!
//! Run: `FT_INV_GOLDEN=1 cargo run -q -j1 -p ft-kernel-cpu --example inv_golden | sha256sum`

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::inv_tensor_contiguous_f64;

fn dump(n: usize) {
    // xorshift64 well-conditioned fill + diagonal dominance so the inverse is
    // well-defined and the factorization is stable.
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15 ^ (n as u64);
    let mut a = vec![0.0f64; n * n];
    for x in a.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *x = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    }
    for i in 0..n {
        a[i * n + i] += n as f64;
    }
    let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
    let inv = match inv_tensor_contiguous_f64(&a, &meta) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("inv_golden n={n} failed: {e:?}");
            std::process::exit(1);
        }
    };
    println!("inv n={n}");
    for (i, v) in inv.iter().enumerate() {
        println!("{i}={:016x}", v.to_bits());
    }
}

fn main() {
    if std::env::var_os("FT_INV_GOLDEN").is_none() {
        eprintln!("set FT_INV_GOLDEN=1 to emit the golden");
        return;
    }
    dump(96);
    dump(256);
    dump(300);
}
