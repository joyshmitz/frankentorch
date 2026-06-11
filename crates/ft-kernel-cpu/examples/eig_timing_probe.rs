//! geev cost-breakdown probe (frankentorch-l9xod / fql10). Times `eig` vs
//! `eigvals` (so `eig - eigvals` = the eigenvector machinery) across n, in ONE
//! process. Run under `RAYON_NUM_THREADS=1` vs default for a same-worker
//! serial/parallel A/B.
//!
//!   rch exec -- cargo run --release -q -p ft-kernel-cpu --example eig_timing_probe
//!
//! Finding (64-thread worker): eig is ~75% `eigvals` (the SERIAL single-bulge
//! Francis QR, O(n^3) — 2592ms@1024) and ~25% eigenvector machinery, the latter
//! ALREADY parallel (q_acc replay + Phase-2 Z·U gemm). The hqr2 back-substitution
//! is only ~3% (parallelizing it REGRESSES). ⇒ the sole geev lever is the
//! multishift-QR + AED rewrite of the Francis QR (qglh3 → npxbw).

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{eig_contiguous_f64, eigvals_contiguous_f64};
use std::time::Instant;
fn build(n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = ((i * 41 + j * 13 + 5) % 17) as f64 * 0.01 - 0.08;
        }
        a[i * n + i] = (i as f64) + 1.0;
    }
    a
}
fn bench<F: FnMut()>(mut f: F, it: usize) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..it {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / it as f64
}
fn main() {
    println!("threads={}", rayon::current_num_threads());
    for &n in &[128usize, 256, 512, 1024] {
        let a = build(n);
        let m = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let it = if n <= 256 {
            8
        } else if n <= 512 {
            4
        } else {
            2
        };
        let ev = bench(
            || {
                let _ = eigvals_contiguous_f64(&a, &m).unwrap();
            },
            it,
        );
        let eg = bench(
            || {
                let _ = eig_contiguous_f64(&a, &m).unwrap();
            },
            it,
        );
        println!(
            "n={n:<5} eigvals={ev:8.2}ms  eig={eg:8.2}ms  (vec_machinery={:.2}ms)",
            eg - ev
        );
    }
}
