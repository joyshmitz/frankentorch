//! Feasibility probe for the packed-panel GEMM (bead frankentorch-z6sjf): can a
//! hand-written safe-Rust register-blocked microkernel match matrixmultiply's per-core
//! throughput? If not (per-core << matrixmultiply's ~52 GF/s/core f64), then even
//! perfect multi-core scaling would net a loss and the rewrite is not worth it.
//! Compares a naive triple loop, an mr x nr register-blocked microkernel (straight-k
//! accumulation, so tolerance-parity-safe), and the production matmul — all SINGLE
//! THREAD (per-core). 1024^3.
//!   cargo run -q --release -p ft-kernel-cpu --example gemm_microkernel_probe
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::matmul_tensor_contiguous_f64;
use std::time::Instant;

#[inline(never)]
fn naive(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

#[inline(never)]
fn blocked(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    const MR: usize = 4;
    const NR: usize = 8;
    let mut c = vec![0.0; m * n];
    let mut i0 = 0;
    while i0 < m {
        let ib = (i0 + MR).min(m) - i0;
        let mut j0 = 0;
        while j0 < n {
            let jb = (j0 + NR).min(n) - j0;
            let mut acc = [[0.0f64; NR]; MR];
            for p in 0..k {
                let brow = &b[p * n + j0..p * n + j0 + jb];
                for ii in 0..ib {
                    let av = a[(i0 + ii) * k + p];
                    let arow = &mut acc[ii];
                    for jj in 0..jb {
                        arow[jj] += av * brow[jj];
                    }
                }
            }
            for ii in 0..ib {
                for jj in 0..jb {
                    c[(i0 + ii) * n + j0 + jj] = acc[ii][jj];
                }
            }
            j0 += NR;
        }
        i0 += MR;
    }
    c
}

fn gflops(m: usize, k: usize, n: usize, ms: f64) -> f64 {
    2.0 * (m as f64) * (k as f64) * (n as f64) / (ms / 1e3) / 1e9
}

fn main() {
    let (m, k, n) = (1024usize, 1024usize, 1024usize);
    let a: Vec<f64> = (0..m * k).map(|i| (i % 101) as f64 * 0.01).collect();
    let b: Vec<f64> = (0..k * n).map(|i| (i % 103) as f64 * 0.01).collect();
    let am = TensorMeta::from_shape(vec![m, k], DType::F64, Device::Cpu);
    let bm = TensorMeta::from_shape(vec![k, n], DType::F64, Device::Cpu);

    // Force single-threaded for a per-core comparison.
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();

    let bench = |f: &dyn Fn() -> Vec<f64>| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            let r = std::hint::black_box(f());
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&r);
        }
        best
    };

    let bn = bench(&|| naive(&a, &b, m, k, n));
    let bb = bench(&|| blocked(&a, &b, m, k, n));
    let bp = pool1.install(|| bench(&|| matmul_tensor_contiguous_f64(&a, &b, &am, &bm).unwrap()));

    // correctness sanity vs naive (tolerance)
    let rn = naive(&a, &b, m, k, n);
    let rb = blocked(&a, &b, m, k, n);
    let maxdiff = rn
        .iter()
        .zip(rb.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);

    eprintln!("gemm 1024^3 (1 core):");
    eprintln!(
        "  naive triple loop : {bn:.1} ms  {:.1} GF/s",
        gflops(m, k, n, bn)
    );
    eprintln!(
        "  blocked 4x8       : {bb:.1} ms  {:.1} GF/s  (blocked-vs-naive maxdiff {maxdiff:.2e})",
        gflops(m, k, n, bb)
    );
    eprintln!(
        "  matrixmultiply    : {bp:.1} ms  {:.1} GF/s",
        gflops(m, k, n, bp)
    );
    eprintln!("  blocked/matrixmultiply per-core ratio: {:.2}x", bp / bb);
}
