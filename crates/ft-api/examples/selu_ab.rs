//! Inference (no-grad) before/after for the selu forward, plus a bit-exact grad
//! check for the parallelized backward.
//! OLD: serial `.iter().map(..exp_m1..)` PLUS the save_for_backward clone the
//! forward ran unconditionally (1x numel f64).
//! NEW: production no-grad path — elements mapped over Rayon AND the
//! backward-only clone skipped. Forward and grad asserted bit-for-bit == serial.
//!   cargo run -q --release -p ft-api --example selu_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

const ALPHA: f64 = 1.6732632423543772848170429916717;
const SCALE: f64 = 1.0507009873554804934193349852946;

fn selu_fwd(x: f64) -> f64 {
    if x > 0.0 {
        SCALE * x
    } else {
        SCALE * ALPHA * x.exp_m1()
    }
}
fn selu_grad(x: f64) -> f64 {
    if x > 0.0 {
        SCALE
    } else {
        SCALE * ALPHA * x.exp()
    }
}

fn old_serial_with_save(x: &[f64]) -> Vec<f64> {
    let values: Vec<f64> = x.iter().map(|&xi| selu_fwd(xi)).collect();
    let _saved = std::hint::black_box(x.to_vec());
    values
}

fn main() {
    let nthreads = rayon::current_num_threads();
    let n = 4_000_000usize;
    // Mixed-sign, negative-leaning so most lanes exercise the exp_m1 branch.
    let x: Vec<f64> = (0..n).map(|i| ((i % 311) as f64) * 0.01 - 2.5).collect();
    let want = old_serial_with_save(&x);
    let want_grad: Vec<f64> = x.iter().map(|&xi| selu_grad(xi)).collect();

    let mut old = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        std::hint::black_box(old_serial_with_save(&x));
        old = old.min(t.elapsed().as_secs_f64() * 1e3);
    }

    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    let new = pn.install(|| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let v = s.tensor_variable(x.clone(), vec![n], false).unwrap();
        let r = s.tensor_selu(v).unwrap();
        let got = s.tensor_values(r).unwrap();
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.to_bits(), w.to_bits(), "selu parallel != serial");
        }
        // Grad path: parallel backward (par_zip_map) must be bit-for-bit == serial.
        let vg = s.tensor_variable(x.clone(), vec![n], true).unwrap();
        let rg = s.tensor_selu(vg).unwrap();
        let loss = s.tensor_sum(rg).unwrap();
        let rep = s.tensor_backward(loss).unwrap();
        let grad = s.tensor_gradient(&rep, vg).expect("selu grad");
        assert_eq!(grad.len(), want_grad.len());
        for (g, w) in grad.iter().zip(want_grad.iter()) {
            assert_eq!(g.to_bits(), w.to_bits(), "selu grad parallel != serial");
        }
        let mut best = f64::INFINITY;
        for _ in 0..15 {
            let v = s.tensor_variable(x.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let r = s.tensor_selu(v).unwrap();
            std::hint::black_box(r);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    });
    println!(
        "selu no-grad n={n} (fwd+grad bit-exact OK): OLD serial+save {old:.2}ms  NEW({nthreads}t) {new:.2}ms  =>  {:.2}x",
        old / new
    );
}
