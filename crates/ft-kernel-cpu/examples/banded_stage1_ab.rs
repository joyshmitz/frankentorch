//! Same-worker A/B for the stage-1 dense->banded reduction (frankentorch-5oqum).
//!
//! Times the new BLAS-3 blocked values-only `symmetric_to_banded_values_f64`
//! against the unblocked reference `symmetric_to_banded_f64` (the band route the
//! two-stage eigvalsh used before), in ONE process on the same pre-built input.
//! Both produce the same band to ~1e-9 (proven by
//! `symmetric_to_banded_values_matches_unblocked`); this measures the
//! O(n^3) BLAS-2 -> level-3 GEMM speedup of stage 1.
//!
//! Run: `cargo run -p ft-kernel-cpu --release --example banded_stage1_ab`

use ft_kernel_cpu::{symmetric_to_banded_f64, symmetric_to_banded_values_f64};
use std::time::Instant;

fn build(n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let val = ((i * 13 + j * 7 + 2) % 31) as f64 * 0.29 - 4.7 + (i as f64) * 0.011
                - (j as f64) * 0.017;
            a[i * n + j] = val;
            a[j * n + i] = val;
        }
    }
    a
}

fn time_it(label: &str, iters: usize, mut f: impl FnMut() -> f64) -> f64 {
    let mut sink = 0.0;
    for _ in 0..3 {
        sink += f();
    }
    let t = Instant::now();
    for _ in 0..iters {
        sink += f();
    }
    let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("  {label:<28} {us:>10.2} us/iter   (sink={sink:.3e})");
    us
}

fn main() {
    for &(n, b) in &[(128usize, 16usize), (256, 32), (512, 32)] {
        let a = build(n);
        let iters = if n <= 128 {
            200
        } else if n <= 256 {
            80
        } else {
            25
        };
        println!("n={n} b={b}  ({iters} iters):");
        let us_new = time_it("blocked values-only (new)", iters, || {
            let band = symmetric_to_banded_values_f64(&a, n, b).unwrap();
            band[n * n - 1]
        });
        let us_old = time_it("unblocked + Q (old)", iters, || {
            let (band, _q) = symmetric_to_banded_f64(&a, n, b).unwrap();
            band[n * n - 1]
        });
        println!("  => speedup {:.2}x\n", us_old / us_new);
    }
}
