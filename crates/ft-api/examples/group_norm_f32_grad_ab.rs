//! End-to-end A/B for the f32 GroupNorm GRAD fast path (frankentorch-48w0b). OLD =
//! the composed op-graph (reshape/mean/expand/sub/sq/var/eps/sqrt/div/reshape +
//! per-channel affine, ~15 materialized ops — the path f32 GroupNorm grad took
//! before; the f64 grad path was already fused but f32 fell through); NEW =
//! functional_group_norm (fused f32-output custom op: group_norm_forward_f32 +
//! group_norm_backward_f32); SCALAR = functional_group_norm_sum, specialized for
//! sum-loss training. Fresh session/iter (tape never frees — gmuml).
//!   cargo run -q --release -p ft-api --example group_norm_f32_grad_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    // CNN GroupNorm: [N, C, H, W].
    let (n, c, h, w, groups) = (8usize, 64usize, 28usize, 28usize, 32usize);
    let sp = h * w;
    let cpg = c / groups;
    let gnum = cpg * sp;
    let eps = 1e-5;
    let xv: Vec<f32> = (0..n * c * sp)
        .map(|i| ((i % 877) as f32 - 400.0) * 0.002)
        .collect();
    let wv: Vec<f32> = (0..c).map(|j| 1.0 + (j % 13) as f32 * 0.01).collect();
    let bvv: Vec<f32> = (0..c).map(|j| (j % 7) as f32 * 0.01 - 0.03).collect();

    let new_step = || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(xv.clone(), vec![n, c, h, w], true)
            .unwrap();
        let wt = s.tensor_variable_f32(wv.clone(), vec![c], true).unwrap();
        let bt = s.tensor_variable_f32(bvv.clone(), vec![c], true).unwrap();
        let o = s
            .functional_group_norm(x, groups, Some(wt), Some(bt), eps)
            .unwrap();
        let l = s.tensor_sum(o).unwrap();
        s.tensor_backward(l).unwrap();
    };
    let scalar_step = || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(xv.clone(), vec![n, c, h, w], true)
            .unwrap();
        let wt = s.tensor_variable_f32(wv.clone(), vec![c], true).unwrap();
        let bt = s.tensor_variable_f32(bvv.clone(), vec![c], true).unwrap();
        let l = s
            .functional_group_norm_sum(x, groups, Some(wt), Some(bt), eps)
            .unwrap();
        s.tensor_backward(l).unwrap();
    };
    // OLD: manual composed GroupNorm op-graph.
    let old_step = || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(xv.clone(), vec![n, c, h, w], true)
            .unwrap();
        let wt = s.tensor_variable_f32(wv.clone(), vec![c], true).unwrap();
        let bt = s.tensor_variable_f32(bvv.clone(), vec![c], true).unwrap();
        let r = s.tensor_reshape(x, vec![n, groups, gnum]).unwrap();
        let mean = s.tensor_mean_dim(r, 2).unwrap();
        let mean = s.tensor_unsqueeze(mean, 2).unwrap();
        let mean = s.tensor_expand(mean, vec![n, groups, gnum]).unwrap();
        let diff = s.tensor_sub(r, mean).unwrap();
        let dsq = s.tensor_mul(diff, diff).unwrap();
        let var = s.tensor_mean_dim(dsq, 2).unwrap();
        let var = s.tensor_unsqueeze(var, 2).unwrap();
        let var = s.tensor_expand(var, vec![n, groups, gnum]).unwrap();
        let epst = s.full(vec![n, groups, gnum], eps, false).unwrap();
        let std = s.tensor_add(var, epst).unwrap();
        let std = s.tensor_sqrt(std).unwrap();
        let norm = s.tensor_div(diff, std).unwrap();
        let norm = s.tensor_reshape(norm, vec![n, c, h, w]).unwrap();
        let we = s.tensor_unsqueeze(wt, 0).unwrap();
        let we = s.tensor_unsqueeze(we, 2).unwrap();
        let we = s.tensor_unsqueeze(we, 3).unwrap();
        let we = s.tensor_expand(we, vec![n, c, h, w]).unwrap();
        let be = s.tensor_unsqueeze(bt, 0).unwrap();
        let be = s.tensor_unsqueeze(be, 2).unwrap();
        let be = s.tensor_unsqueeze(be, 3).unwrap();
        let be = s.tensor_expand(be, vec![n, c, h, w]).unwrap();
        let o = s.tensor_mul(we, norm).unwrap();
        let o = s.tensor_add(o, be).unwrap();
        let l = s.tensor_sum(o).unwrap();
        s.tensor_backward(l).unwrap();
    };

    new_step();
    scalar_step();
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
    let mut bs = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        scalar_step();
        bs = bs.min(t.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!(
        "group_norm f32 fwd+bwd [{n},{c},{h}x{w}] g{groups}: composed {bo:.2} ms / fused {bn:.2} ms / scalar_sum {bs:.2} ms / fused_speedup {:.2}x / scalar_vs_fused {:.2}x",
        bo / bn,
        bn / bs
    );
}
