//! Anchored A/B for logsumexp BACKWARD lane parallelization (grad path).
//! grad_x[idx] = g * exp(x[idx]-y), one exp/element, per-lane independent = compute-bound.
//! Replicates the production backward math serial vs parallel, 1t-vs-Nt in one process,
//! asserts bit-for-bit equality.
//!   cargo run -q --release -p ft-api --example logsumexp_bwd_ab
use rayon::prelude::*;
use std::time::Instant;

fn lane(outer: usize, out: &mut [f64], x: &[f64], y: &[f64], gy: &[f64], dim: usize, inner: usize) {
    let base = outer * dim * inner;
    for i in 0..inner {
        let yidx = outer * inner + i;
        let yv = y[yidx];
        let g = gy[yidx];
        for d in 0..dim {
            let l = d * inner + i;
            out[l] = g * (x[base + l] - yv).exp();
        }
    }
}

fn run_best(
    threads: usize,
    x: &[f64],
    y: &[f64],
    gy: &[f64],
    outer: usize,
    dim: usize,
    inner: usize,
) -> (f64, Vec<f64>) {
    let ln = dim * inner;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let run = || {
            let mut g = vec![0.0; x.len()];
            if threads > 1 {
                g.par_chunks_mut(ln)
                    .enumerate()
                    .for_each(|(o, c)| lane(o, c, x, y, gy, dim, inner));
            } else {
                for o in 0..outer {
                    let b = o * ln;
                    lane(o, &mut g[b..b + ln], x, y, gy, dim, inner);
                }
            }
            g
        };
        let out = run();
        let mut best = f64::INFINITY;
        for _ in 0..15 {
            let t = Instant::now();
            std::hint::black_box(run());
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        (best, out)
    })
}

fn main() {
    let (rows, cols) = (16384usize, 256);
    let n = rows * cols;
    let x: Vec<f64> = (0..n).map(|i| ((i % 211) as f64) * 0.01 - 1.0).collect();
    // y = logsumexp over dim=1 -> [rows]; compute a plausible per-row reduction.
    let mut y = vec![0.0; rows];
    for o in 0..rows {
        let mut m = f64::NEG_INFINITY;
        for d in 0..cols {
            m = m.max(x[o * cols + d]);
        }
        let mut s = 0.0;
        for d in 0..cols {
            s += (x[o * cols + d] - m).exp();
        }
        y[o] = m + s.ln();
    }
    let gy: Vec<f64> = (0..rows).map(|i| 1.0 + ((i % 13) as f64) * 1e-3).collect();
    let nthreads = rayon::current_num_threads();

    let (ser, gs) = run_best(1, &x, &y, &gy, rows, cols, 1);
    let (par, gp) = run_best(nthreads.max(2), &x, &y, &gy, rows, cols, 1);
    for (a, b) in gs.iter().zip(gp.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "parallel != serial bit-exact");
    }
    println!(
        "logsumexp_backward [{rows},{cols}] dim=1 (bit-exact OK): serial(1t) {ser:.2}ms  parallel({nthreads}t) {par:.2}ms  =>  {:.2}x",
        ser / par
    );
}
