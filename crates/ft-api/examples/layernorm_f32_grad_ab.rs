//! End-to-end A/B for the f32 LayerNorm GRAD fast path (frankentorch-48w0b). OLD =
//! the composed op-graph (mean/var/normalize/affine, ~14 materialized ops) the
//! f32 grad path took before; NEW = functional_layer_norm (fused f32-output
//! custom op: layer_norm_forward_f32 + layer_norm_backward_f32). Fresh session/iter.
//!   cargo run -q --release -p ft-api --example layernorm_f32_grad_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    // Transformer LayerNorm: [batch*seq, d_model].
    let (rows, nn) = (8192usize, 1024usize);
    let eps = 1e-5;
    let xv: Vec<f32> = (0..rows * nn).map(|i| ((i % 877) as f32 - 400.0) * 0.002).collect();
    let wv: Vec<f32> = (0..nn).map(|j| 1.0 + (j % 13) as f32 * 0.01).collect();
    let bvv: Vec<f32> = (0..nn).map(|j| (j % 7) as f32 * 0.01 - 0.03).collect();

    let new_step = || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable_f32(xv.clone(), vec![rows, nn], true).unwrap();
        let w = s.tensor_variable_f32(wv.clone(), vec![nn], true).unwrap();
        let b = s.tensor_variable_f32(bvv.clone(), vec![nn], true).unwrap();
        let o = s.functional_layer_norm(x, vec![nn], Some(w), Some(b), eps).unwrap();
        let l = s.tensor_sum(o).unwrap();
        s.tensor_backward(l).unwrap();
    };
    // OLD: manual composed LayerNorm (mean/var/normalize/affine), as the op-graph.
    let old_step = || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable_f32(xv.clone(), vec![rows, nn], true).unwrap();
        let w = s.tensor_variable_f32(wv.clone(), vec![nn], true).unwrap();
        let b = s.tensor_variable_f32(bvv.clone(), vec![nn], true).unwrap();
        let mean = s.tensor_mean_dim(x, 1).unwrap();
        let mean = s.tensor_unsqueeze(mean, 1).unwrap();
        let mean = s.tensor_expand(mean, vec![rows, nn]).unwrap();
        let diff = s.tensor_sub(x, mean).unwrap();
        let dsq = s.tensor_mul(diff, diff).unwrap();
        let var = s.tensor_mean_dim(dsq, 1).unwrap();
        let var = s.tensor_unsqueeze(var, 1).unwrap();
        let var = s.tensor_expand(var, vec![rows, nn]).unwrap();
        let epst = s.full(vec![rows, nn], eps, false).unwrap();
        let std = s.tensor_add(var, epst).unwrap();
        let std = s.tensor_sqrt(std).unwrap();
        let norm = s.tensor_div(diff, std).unwrap();
        let we = s.tensor_unsqueeze(w, 0).unwrap();
        let we = s.tensor_expand(we, vec![rows, nn]).unwrap();
        let be = s.tensor_unsqueeze(b, 0).unwrap();
        let be = s.tensor_expand(be, vec![rows, nn]).unwrap();
        let o = s.tensor_mul(we, norm).unwrap();
        let o = s.tensor_add(o, be).unwrap();
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
    eprintln!("layernorm f32 fwd+bwd [{rows}x{nn}]: composed {bo:.2} ms / fused {bn:.2} ms / speedup {:.2}x", bo / bn);
}
