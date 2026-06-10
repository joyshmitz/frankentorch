//! Quantify the two-stage band-reduction eigvalsh win vs the live (direct tred2)
//! path at large n, to size the tolerance-parity policy unlock (frankentorch-qgce4).
//!
//! Run: cargo run --release -j1 -p ft-kernel-cpu --example two_stage_eig_quant

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{eigvalsh_contiguous_f64, eigvalsh_two_stage_f64};
use std::time::Instant;

fn sym_matrix(n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let bij = ((i * 31 + j * 17) % 97) as f64 * 0.013 - 0.5;
            let bji = ((j * 31 + i * 17) % 97) as f64 * 0.013 - 0.5;
            a[i * n + j] = 0.5 * (bij + bji);
        }
        a[i * n + i] += n as f64;
    }
    a
}

fn bench<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    f(); // warm
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn main() {
    for &(n, b) in &[(256usize, 32usize), (512, 32), (768, 48), (1024, 48)] {
        let a = sym_matrix(n);
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let iters = if n <= 512 { 5 } else { 2 };
        let live = bench(
            || {
                let _ = eigvalsh_contiguous_f64(&a, &meta).unwrap();
            },
            iters,
        );
        let two = bench(
            || {
                let _ = eigvalsh_two_stage_f64(&a, n, b).unwrap();
            },
            iters,
        );
        // Max abs eigenvalue diff (sorted) to confirm tolerance parity.
        let mut l = eigvalsh_contiguous_f64(&a, &meta).unwrap();
        let mut t = eigvalsh_two_stage_f64(&a, n, b).unwrap();
        l.sort_by(f64::total_cmp);
        t.sort_by(f64::total_cmp);
        let maxdiff = l
            .iter()
            .zip(&t)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
        println!(
            "n={n:<5} b={b:<3} live={live:8.3}ms two_stage={two:8.3}ms  ratio={:.2}x  maxdiff={maxdiff:.2e}",
            live / two
        );
    }
}
