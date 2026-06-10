//! Deterministic SVD golden for the Golub-Reinsch path (frankentorch-svd perf).
//!
//! Emits the bit pattern of every U / S / V^T entry for both the
//! deferred-left fast path (`full_matrices=false`, well-conditioned square) and
//! the full track-left sweep (`full_matrices=true`). Piping this through
//! `sha256sum` gives a before/after isomorphism proof for sweep-restructuring
//! levers that must stay bit-for-bit identical.
//!
//! Run: `FT_SVD_GOLDEN=1 cargo run -q -j1 -p ft-kernel-cpu --example svd_golden | sha256sum`

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::svd_contiguous_f64;

fn deterministic_matrix(n: usize, seed: u64) -> Vec<f64> {
    // xorshift64 well-conditioned fill (same family as the bench `_wellcond`).
    let mut s = seed;
    let mut a = vec![0.0f64; n * n];
    for x in a.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *x = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    }
    a
}

fn dump(label: &str, n: usize, full: bool) {
    let a = deterministic_matrix(n, 0x9e37_79b9_7f4a_7c15);
    let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
    let r = match svd_contiguous_f64(&a, &meta, full) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("svd_golden {label} failed: {e:?}");
            std::process::exit(1);
        }
    };
    println!("{label} m={} n={} k={}", r.m, r.n, r.k);
    for (i, v) in r.s.iter().enumerate() {
        println!("S[{i}]={:016x}", v.to_bits());
    }
    for (i, v) in r.u.iter().enumerate() {
        println!("U[{i}]={:016x}", v.to_bits());
    }
    for (i, v) in r.vh.iter().enumerate() {
        println!("Vh[{i}]={:016x}", v.to_bits());
    }
}

fn main() {
    if std::env::var_os("FT_SVD_GOLDEN").is_none() {
        eprintln!("set FT_SVD_GOLDEN=1 to emit the golden");
        return;
    }
    // n>=64 well-conditioned square exercises the deferred-left fast path.
    dump("deferred_left_96", 96, false);
    dump("deferred_left_128", 128, false);
    // full_matrices=true forces the full track-left golub_reinsch sweep.
    dump("full_sweep_96", 96, true);
    dump("full_sweep_128", 128, true);
}
