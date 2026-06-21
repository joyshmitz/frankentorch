//! End-to-end A/B for the f32 SDPA (attention) GRAD fast path (frankentorch-48w0b).
//! OLD = composed bmm + scale + softmax + bmm (what f32 attention grad took
//! before); NEW = scaled_dot_product_attention (fused f32-output custom op:
//! sdpa_forward_f32 + sdpa_backward_f32). Fresh session per iter.
//!   cargo run -q --release -p ft-api --example sdpa_f32_grad_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    // [num_bh = batch*heads, seq, d] transformer attention.
    let (bh, seq, d) = (32usize, 128usize, 64usize);
    let n = bh * seq * d;
    let qv: Vec<f32> = (0..n).map(|i| ((i % 877) as f32 - 400.0) * 0.001).collect();
    let kv: Vec<f32> = (0..n).map(|i| ((i % 691) as f32 - 300.0) * 0.001).collect();
    let vv: Vec<f32> = (0..n).map(|i| ((i % 599) as f32 - 250.0) * 0.001).collect();
    let scale = 1.0 / (d as f64).sqrt();

    let new_step = || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let q = s
            .tensor_variable_f32(qv.clone(), vec![bh, seq, d], true)
            .unwrap();
        let k = s
            .tensor_variable_f32(kv.clone(), vec![bh, seq, d], true)
            .unwrap();
        let v = s
            .tensor_variable_f32(vv.clone(), vec![bh, seq, d], true)
            .unwrap();
        let o = s
            .scaled_dot_product_attention(q, k, v, None, 0.0, false)
            .unwrap();
        let l = s.tensor_sum(o).unwrap();
        s.tensor_backward(l).unwrap();
    };
    let old_step = || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let q = s
            .tensor_variable_f32(qv.clone(), vec![bh, seq, d], true)
            .unwrap();
        let k = s
            .tensor_variable_f32(kv.clone(), vec![bh, seq, d], true)
            .unwrap();
        let v = s
            .tensor_variable_f32(vv.clone(), vec![bh, seq, d], true)
            .unwrap();
        let kt = s.tensor_transpose(k, 1, 2).unwrap();
        let aw = s.tensor_bmm(q, kt).unwrap();
        let sc = s.full(vec![bh, seq, seq], scale, false).unwrap();
        let scaled = s.tensor_mul(aw, sc).unwrap();
        let p = s.tensor_softmax(scaled, 2).unwrap();
        let o = s.tensor_bmm(p, v).unwrap();
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
        "sdpa f32 fwd+bwd [{bh},{seq},{d}]: composed(bmm+softmax+bmm) {bo:.2} ms / fused {bn:.2} ms / speedup {:.2}x",
        bo / bn
    );
}
