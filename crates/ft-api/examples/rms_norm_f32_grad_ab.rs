//! End-to-end A/B for the f32 RMSNorm GRAD fast path (frankentorch-48w0b). OLD =
//! the composed op-graph (square/mean/eps/rsqrt/normalize/affine, ~9 materialized
//! ops — the path RMSNorm grad took before; the f64 kernel existed but was never
//! wired); NEW = functional_rms_norm (fused f32-output custom op:
//! rms_norm_forward_f32 + rms_norm_backward_f32). Fresh session/iter (tape never
//! frees — gmuml). Min-time per arm.
//!   cargo run -q --release -p ft-api --example rms_norm_f32_grad_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    // LLaMA-style RMSNorm: [batch*seq, d_model].
    let (rows, nn) = (8192usize, 1024usize);
    let eps = 1e-6;
    let xv: Vec<f32> = (0..rows * nn)
        .map(|i| ((i % 877) as f32 - 400.0) * 0.002)
        .collect();
    let wv: Vec<f32> = (0..nn).map(|j| 1.0 + (j % 13) as f32 * 0.01).collect();

    let new_step = || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(xv.clone(), vec![rows, nn], true)
            .unwrap();
        let w = s.tensor_variable_f32(wv.clone(), vec![nn], true).unwrap();
        let o = s.functional_rms_norm(x, vec![nn], Some(w), eps).unwrap();
        let l = s.tensor_sum(o).unwrap();
        s.tensor_backward(l).unwrap();
    };
    // OLD: manual composed RMSNorm (square/mean/eps/rsqrt/normalize/affine).
    let old_step = || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(xv.clone(), vec![rows, nn], true)
            .unwrap();
        let w = s.tensor_variable_f32(wv.clone(), vec![nn], true).unwrap();
        let sq = s.tensor_mul(x, x).unwrap();
        let ms = s.tensor_mean_dim(sq, 1).unwrap();
        let ms = s.tensor_unsqueeze(ms, 1).unwrap();
        let ms = s.tensor_expand(ms, vec![rows, nn]).unwrap();
        let epst = s.full(vec![rows, nn], eps, false).unwrap();
        let rms = s.tensor_add(ms, epst).unwrap();
        let rms = s.tensor_sqrt(rms).unwrap();
        let norm = s.tensor_div(x, rms).unwrap();
        let we = s.tensor_unsqueeze(w, 0).unwrap();
        let we = s.tensor_expand(we, vec![rows, nn]).unwrap();
        let o = s.tensor_mul(we, norm).unwrap();
        let l = s.tensor_sum(o).unwrap();
        s.tensor_backward(l).unwrap();
    };

    new_step();
    old_step();
    let reps = 12;
    let mut bo = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        old_step();
        bo = bo.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut bn = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        new_step();
        bn = bn.min(t.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!(
        "rms_norm f32 fwd+bwd [{rows}x{nn}]: composed {bo:.2} ms / fused {bn:.2} ms / speedup {:.2}x",
        bo / bn
    );
}
