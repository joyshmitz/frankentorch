//! Single-arm timing of Rprop.step (default rayon pool, op-only, fresh session/iter,
//! 2M params). Used as the OLD baseline before fusing Rprop, then re-run after the
//! refactor to measure the memory-pass-reduction (fusion) win old-vs-new.
//!   cargo run -q --release -p ft-optim --example rprop_step_time
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use ft_optim::{Optimizer, Rprop};
use std::time::Instant;

fn main() {
    let n = 1usize << 21; // 2M params
    let mut best = f64::INFINITY;
    for _ in 0..20 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![0.5_f64; n], vec![n], true).unwrap();
        let sq = s.tensor_mul(x, x).unwrap();
        let loss = s.tensor_sum(sq).unwrap();
        let report = s.tensor_backward(loss).unwrap();
        let mut opt = Rprop::new(vec![x], 0.01);
        let t = Instant::now();
        opt.step(&mut s, &report).unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!("rprop step [{n} params]: {best:.3} ms");
}
