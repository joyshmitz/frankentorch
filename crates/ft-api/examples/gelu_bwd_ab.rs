//! Before/after for the compute-bound activation BACKWARD contrib maps in
//! ft-autograd's TensorTape backward (gelu representative; silu/elu/erf/erfc/
//! softplus/mish share the structure). The backward `contrib` was a serial
//! `incoming.iter().zip(x).map(<transcendental grad>).collect()`; it now fans
//! over Rayon (par_zip_map_grad, order-preserving => bit-for-bit identical).
//!
//! OLD proxy = the serial contrib map (the dominant old-backward work).
//! NEW = production `tensor_backward` on a single-gelu graph (parallel contrib +
//! small graph/accumulate overhead, so the measured ratio is conservative).
//! Grad asserted bit-for-bit == serial reference.
//!   cargo run -q --release -p ft-api --example gelu_bwd_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn gelu_grad(x: f64) -> f64 {
    let inv_sqrt_two = std::f64::consts::FRAC_1_SQRT_2;
    let inv_sqrt_two_pi = std::f64::consts::FRAC_1_SQRT_2 * std::f64::consts::FRAC_2_SQRT_PI * 0.5;
    let phi = inv_sqrt_two_pi * (-0.5 * x * x).exp();
    0.5 * (1.0 + libm::erf(x * inv_sqrt_two)) + x * phi
}

fn old_serial_contrib(incoming: &[f64], x: &[f64]) -> Vec<f64> {
    incoming
        .iter()
        .zip(x.iter())
        .map(|(&g, &xi)| g * gelu_grad(xi))
        .collect()
}

fn main() {
    let nthreads = rayon::current_num_threads();
    let n = 4_000_000usize;
    let x: Vec<f64> = (0..n).map(|i| ((i % 401) as f64) * 0.01 - 2.0).collect();
    let ones = vec![1.0f64; n];
    let want_grad = old_serial_contrib(&ones, &x);

    let mut old = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        std::hint::black_box(old_serial_contrib(&ones, &x));
        old = old.min(t.elapsed().as_secs_f64() * 1e3);
    }

    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    let new = pn.install(|| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Correctness: parallel backward grad must be bit-for-bit == serial.
        let vg = s.tensor_variable(x.clone(), vec![n], true).unwrap();
        let y = s.tensor_gelu(vg).unwrap();
        let loss = s.tensor_sum(y).unwrap();
        let rep = s.tensor_backward(loss).unwrap();
        let grad = s.tensor_gradient(&rep, vg).expect("gelu grad");
        assert_eq!(grad.len(), want_grad.len());
        for (g, w) in grad.iter().zip(want_grad.iter()) {
            assert_eq!(g.to_bits(), w.to_bits(), "gelu backward parallel != serial");
        }
        let mut best = f64::INFINITY;
        for _ in 0..15 {
            let vg = s.tensor_variable(x.clone(), vec![n], true).unwrap();
            let y = s.tensor_gelu(vg).unwrap();
            let loss = s.tensor_sum(y).unwrap();
            let t = Instant::now();
            let rep = s.tensor_backward(loss).unwrap();
            std::hint::black_box(s.tensor_gradient(&rep, vg));
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    });
    println!(
        "gelu backward n={n} (grad bit-exact OK): OLD serial-contrib {old:.2}ms  NEW({nthreads}t) tensor_backward {new:.2}ms  =>  {:.2}x",
        old / new
    );
}
