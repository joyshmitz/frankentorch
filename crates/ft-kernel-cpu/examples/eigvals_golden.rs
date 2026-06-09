//! Golden-fixture generator for the NON-symmetric `eigvals`/`eig` Francis-QR
//! path (`frankentorch-l9xod`). Prints an order-sensitive bit digest of every
//! eigenvalue (interleaved re,im) for fixed non-symmetric matrices, so the
//! deep multishift-QR + aggressive-early-deflation rewrite can prove its
//! strict-mode (classic double-shift) fallback stays BIT-EXACT — the
//! before/after golden the parity-contract migration needs. Mirrors
//! `eigh_golden.rs` (the symmetric counterpart for `frankentorch-ncwz`).
//!
//!   rch exec -- cargo run -p ft-kernel-cpu --release --example eigvals_golden
//!
//! Recorded golden at HEAD ac46bc36 (eigvals == eig, bit-for-bit):
//!   n=64  : 0xbc0583d464b1a211
//!   n=128 : 0xcf8084e9cc30d867
//!   n=256 : 0x188d322a66b49c0f

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{eig_contiguous_f64, eigvals_contiguous_f64};

fn build(n: usize) -> Vec<f64> {
    // Deterministic non-symmetric fill with a strong diagonal (same fixture as
    // the eig_parallel_schur bit-exact unit test): the spectrum is well
    // separated and the QR sweeps deflate into several sub-blocks.
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = ((i * 41 + j * 13 + 5) % 17) as f64 * 0.01 - 0.08;
        }
        a[i * n + i] = (i as f64) + 1.0;
    }
    a
}

fn fnv1a(values: &[f64]) -> u64 {
    // Order-sensitive 64-bit digest of the exact eigenvalue bit patterns.
    let mut h: u64 = 0xcbf29ce484222325;
    for v in values {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn main() {
    for &n in &[64usize, 128, 256] {
        let a = build(n);
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);

        let vals = match eigvals_contiguous_f64(&a, &meta) {
            Ok(v) => v,
            Err(err) => {
                eprintln!("eigvals_golden failed n={n}: {err}");
                std::process::exit(1);
            }
        };
        // eig (want_vectors=true) shares the QR core; print its eigenvalue digest
        // too so the rewrite can confirm both paths stay locked together.
        let full = match eig_contiguous_f64(&a, &meta) {
            Ok(v) => v,
            Err(err) => {
                eprintln!("eig_golden failed n={n}: {err}");
                std::process::exit(1);
            }
        };

        println!("frankentorch-l9xod eigvals_golden n={n}");
        println!("eigvals_digest={:#018x}", fnv1a(&vals));
        println!("eig_digest={:#018x}", fnv1a(&full.eigenvalues));
    }
}
