//! End-to-end A/B for the f32 BatchNorm2d (training) GRAD fast path
//! (frankentorch-48w0b). OLD = the composed op-graph (permute/stats/normalize/
//! affine/permute + running-stat updates — the path f32 BN grad took before; the
//! f64 grad path was already fused but f32 fell through and upcast to F64); NEW =
//! functional_batch_norm2d (fused f32-output custom op: batch_norm_stats_f32 +
//! batch_norm_apply_f32 fwd, batch_norm_backward_f32 bwd). SCALAR =
//! functional_batch_norm2d_sum (loss-specialized f32 scalar path). Fresh
//! session/iter.
//!   cargo run -q --release -p ft-api --example batch_norm_f32_grad_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    // CNN BatchNorm2d: [N, C, H, W].
    let (n, c, h, w) = (16usize, 64usize, 28usize, 28usize);
    let sp = h * w;
    let m = (n * sp) as f64;
    let eps = 1e-5;
    let mom = 0.1;
    let xv: Vec<f32> = (0..n * c * sp)
        .map(|i| ((i % 877) as f32 - 400.0) * 0.002)
        .collect();
    let wv: Vec<f32> = (0..c).map(|j| 1.0 + (j % 13) as f32 * 0.01).collect();
    let bvv: Vec<f32> = (0..c).map(|j| (j % 7) as f32 * 0.01 - 0.03).collect();
    let rmv: Vec<f32> = (0..c).map(|j| (j % 5) as f32 * 0.02).collect();
    let rvv: Vec<f32> = (0..c).map(|j| 1.0 + (j % 3) as f32 * 0.05).collect();

    let new_step = || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(xv.clone(), vec![n, c, h, w], true)
            .unwrap();
        let rm = s.tensor_variable_f32(rmv.clone(), vec![c], false).unwrap();
        let rv = s.tensor_variable_f32(rvv.clone(), vec![c], false).unwrap();
        let wt = s.tensor_variable_f32(wv.clone(), vec![c], true).unwrap();
        let bt = s.tensor_variable_f32(bvv.clone(), vec![c], true).unwrap();
        let (o, _, _) = s
            .functional_batch_norm2d(x, Some(rm), Some(rv), Some(wt), Some(bt), true, mom, eps)
            .unwrap();
        let l = s.tensor_sum(o).unwrap();
        s.tensor_backward(l).unwrap();
    };
    let scalar_step = || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(xv.clone(), vec![n, c, h, w], true)
            .unwrap();
        let rm = s.tensor_variable_f32(rmv.clone(), vec![c], false).unwrap();
        let rv = s.tensor_variable_f32(rvv.clone(), vec![c], false).unwrap();
        let wt = s.tensor_variable_f32(wv.clone(), vec![c], true).unwrap();
        let bt = s.tensor_variable_f32(bvv.clone(), vec![c], true).unwrap();
        let (l, _, _) = s
            .functional_batch_norm2d_sum(x, Some(rm), Some(rv), Some(wt), Some(bt), true, mom, eps)
            .unwrap();
        s.tensor_backward(l).unwrap();
    };
    // OLD: manual composed BatchNorm2d op-graph (per-channel stats over N,H,W).
    let old_step = || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(xv.clone(), vec![n, c, h, w], true)
            .unwrap();
        let wt = s.tensor_variable_f32(wv.clone(), vec![c], true).unwrap();
        let bt = s.tensor_variable_f32(bvv.clone(), vec![c], true).unwrap();
        // [N,C,H,W] -> [C, N*H*W] via permute(1,0,2,3)+reshape so the per-channel
        // stats reduce over the right axis.
        let perm = s.tensor_permute(x, vec![1, 0, 2, 3]).unwrap();
        let flat = s.tensor_reshape(perm, vec![c, n * sp]).unwrap();
        let mean = s.tensor_mean_dim(flat, 1).unwrap();
        let mean = s.tensor_unsqueeze(mean, 1).unwrap();
        let mean = s.tensor_expand(mean, vec![c, n * sp]).unwrap();
        let diff = s.tensor_sub(flat, mean).unwrap();
        let dsq = s.tensor_mul(diff, diff).unwrap();
        let var = s.tensor_mean_dim(dsq, 1).unwrap();
        let var = s.tensor_unsqueeze(var, 1).unwrap();
        let var = s.tensor_expand(var, vec![c, n * sp]).unwrap();
        let epst = s.full(vec![c, n * sp], eps, false).unwrap();
        let std = s.tensor_add(var, epst).unwrap();
        let std = s.tensor_sqrt(std).unwrap();
        let norm = s.tensor_div(diff, std).unwrap();
        let we = s.tensor_unsqueeze(wt, 1).unwrap();
        let we = s.tensor_expand(we, vec![c, n * sp]).unwrap();
        let be = s.tensor_unsqueeze(bt, 1).unwrap();
        let be = s.tensor_expand(be, vec![c, n * sp]).unwrap();
        let o = s.tensor_mul(we, norm).unwrap();
        let o = s.tensor_add(o, be).unwrap();
        let l = s.tensor_sum(o).unwrap();
        s.tensor_backward(l).unwrap();
    };
    let _ = m;

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
        "batch_norm2d f32 fwd+bwd [{n},{c},{h}x{w}]: composed {bo:.2} ms / fused {bn:.2} ms / scalar {bs:.2} ms / fused speedup {:.2}x / scalar-vs-fused {:.2}x",
        bo / bn,
        bn / bs
    );
}
