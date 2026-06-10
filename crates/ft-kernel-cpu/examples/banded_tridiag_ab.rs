//! Same-worker A/B for the stage-2 band-packed bulge chase (frankentorch-5oqum).
//!
//! Times the live band-packed `banded_to_tridiagonal_f64` against an inlined
//! copy of the original full-length (`0..n`) apply, in ONE process, on the same
//! pre-built band. Both produce bit-identical `(d, e)` (proven by the unit test
//! `banded_to_tridiagonal_band_packed_is_bit_exact`); this measures only the
//! O(n^2 b) vs O(n^3) cost of the rotation sweep.
//!
//! Run: `cargo run -p ft-kernel-cpu --release --example banded_tridiag_ab`

use ft_kernel_cpu::{banded_to_tridiagonal_f64, symmetric_to_banded_f64};
use std::time::Instant;

/// The pre-band-packing reference: full-length `0..n` Givens apply.
fn reference(band: &[f64], n: usize, b: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = band[..n * n].to_vec();
    let apply = |m: &mut [f64], p: usize, q: usize, c: f64, s: f64| {
        for col in 0..n {
            let mp = m[p * n + col];
            let mq = m[q * n + col];
            m[p * n + col] = c * mp + s * mq;
            m[q * n + col] = -s * mp + c * mq;
        }
        for row in 0..n {
            let mp = m[row * n + p];
            let mq = m[row * n + q];
            m[row * n + p] = c * mp + s * mq;
            m[row * n + q] = -s * mp + c * mq;
        }
    };
    if n > 2 {
        for bw in (2..=b.min(n - 1)).rev() {
            for i in 0..n.saturating_sub(bw) {
                let mut tr = i + bw;
                let mut tc = i;
                while tr < n {
                    let piv = m[(tr - 1) * n + tc];
                    let kill = m[tr * n + tc];
                    if kill == 0.0 {
                        break;
                    }
                    let r = piv.hypot(kill);
                    let c = piv / r;
                    let s = kill / r;
                    apply(&mut m, tr - 1, tr, c, s);
                    m[tr * n + tc] = 0.0;
                    m[tc * n + tr] = 0.0;
                    tc = tr - 1;
                    tr += bw;
                }
            }
        }
    }
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n];
    for i in 0..n {
        d[i] = m[i * n + i];
    }
    for i in 1..n {
        e[i] = m[i * n + (i - 1)];
    }
    (d, e)
}

fn build_band(n: usize, b: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let val = ((i * 7 + j * 13 + 5) % 29) as f64 * 0.37 - 5.1 + (i as f64) * 0.03
                - (j as f64) * 0.02;
            a[i * n + j] = val;
            a[j * n + i] = val;
        }
    }
    symmetric_to_banded_f64(&a, n, b).unwrap().0
}

fn time_it(label: &str, iters: usize, mut f: impl FnMut() -> f64) -> f64 {
    // warm-up
    let mut sink = 0.0;
    for _ in 0..3 {
        sink += f();
    }
    let t = Instant::now();
    for _ in 0..iters {
        sink += f();
    }
    let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("  {label:<22} {us:>10.2} us/iter   (sink={sink:.3e})");
    us
}

fn main() {
    for &(n, b) in &[(256usize, 16usize), (256, 32), (512, 32)] {
        let band = build_band(n, b);
        let iters = if n <= 256 { 200 } else { 60 };
        println!("n={n} b={b}  ({iters} iters):");
        // Interleave to share worker/thermal state across both arms.
        let us_packed = time_it("band-packed (new)", iters, || {
            let (d, e) = banded_to_tridiagonal_f64(&band, n, b);
            d[0] + e[n - 1]
        });
        let us_ref = time_it("full-length (old)", iters, || {
            let (d, e) = reference(&band, n, b);
            d[0] + e[n - 1]
        });
        println!("  => speedup {:.2}x\n", us_ref / us_packed);
    }
}
