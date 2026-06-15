//! ISOLATED lever A/B for the transcendental activation BACKWARD contrib maps in
//! ft-autograd (gelu/silu/elu/erf/erfc/softplus/mish). Times JUST the contrib
//! computation `out[i] = g[i] * grad(x[i])` serial vs the order-preserving Rayon
//! map (`par_zip_map_grad` mirror), in ONE process on ONE worker, with cumsum-
//! style anchoring discipline. The grad function is the gelu derivative (the
//! heaviest of the family: erf + exp); the per-element work is what amortizes the
//! fork/join, so the other members share the win.
//!
//! NEW result is asserted BIT-FOR-BIT == serial (the indexed zip preserves order).
//!   cargo run -q --release -p ft-api --example activation_bwd_contrib_ab
use rayon::prelude::*;
use std::time::Instant;

#[inline]
fn gelu_grad(x: f64) -> f64 {
    let inv_sqrt_two = std::f64::consts::FRAC_1_SQRT_2;
    let inv_sqrt_two_pi = std::f64::consts::FRAC_1_SQRT_2 * std::f64::consts::FRAC_2_SQRT_PI * 0.5;
    let phi = inv_sqrt_two_pi * (-0.5 * x * x).exp();
    0.5 * (1.0 + libm::erf(x * inv_sqrt_two)) + x * phi
}

fn serial_contrib(incoming: &[f64], x: &[f64]) -> Vec<f64> {
    incoming
        .iter()
        .zip(x.iter())
        .map(|(&g, &xi)| g * gelu_grad(xi))
        .collect()
}

fn par_contrib(incoming: &[f64], x: &[f64]) -> Vec<f64> {
    incoming
        .par_iter()
        .zip(x.par_iter())
        .map(|(&g, &xi)| g * gelu_grad(xi))
        .collect()
}

fn bench(label: &str, n: usize) {
    let nthreads = rayon::current_num_threads();
    let x: Vec<f64> = (0..n).map(|i| ((i % 401) as f64) * 0.01 - 2.0).collect();
    let incoming: Vec<f64> = (0..n).map(|i| 1.0 + ((i % 7) as f64) * 0.1).collect();

    let want = serial_contrib(&incoming, &x);
    let got = par_contrib(&incoming, &x);
    assert_eq!(want.len(), got.len());
    for (a, b) in want.iter().zip(got.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "parallel contrib != serial");
    }

    let mut old = f64::INFINITY;
    for _ in 0..20 {
        let t = Instant::now();
        std::hint::black_box(serial_contrib(&incoming, &x));
        old = old.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut new = f64::INFINITY;
    for _ in 0..20 {
        let t = Instant::now();
        std::hint::black_box(par_contrib(&incoming, &x));
        new = new.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "{label} n={n} ({nthreads}t, bit-exact OK): serial {old:.3}ms  par {new:.3}ms  =>  {:.2}x",
        old / new
    );
}

fn main() {
    // ANCHOR small shape first (serial should win / par near 1x — flags a bad worker
    // if the big shape also fails to scale), then the production-scale shape.
    bench("ANCHOR-small", 4096);
    bench("contrib", 1_000_000);
    bench("contrib", 4_000_000);
}
