//! conv_transpose2d double-backward probe vs torch (frankentorch-lboou): input
//! Hessian + gradient-penalty (weight+bias). The fused saved-input grad op had
//! no create_graph backward → 2nd-order ERR'd; transposed conv is bilinear so the
//! same generic recipe (CTforward/CTbackward as mutual adjoints) applies.
//!   cargo run -q --release -p ft-api --example conv_transpose_hessian_probe
use ft_api::FrankenTorchSession;
use ft_autograd::BackwardOptions;
use ft_core::ExecutionMode;

fn main() {
    // input-Hessian: x[1,1,2,2], w[1,1,3,3] -> out[1,1,4,4]. torch diag=[1.28;4].
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s
        .tensor_variable((0..4).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect(), vec![1, 1, 2, 2], true)
        .unwrap();
    let w = s
        .tensor_variable((0..9).map(|i| (i % 5) as f64 * 0.2 - 0.4).collect(), vec![1, 1, 3, 3], false)
        .unwrap();
    match (|| {
        let y = s.functional_conv_transpose2d(x, w, None, (1, 1), (0, 0), (0, 0))?;
        let sq = s.tensor_mul(y, y)?;
        let loss = s.tensor_sum(sq)?;
        s.tensor_functional_hessian(loss, x)
    })() {
        Ok(h) => {
            let d: Vec<f64> = (0..4).map(|i| (h[i * 4 + i] * 1e5).round() / 1e5).collect();
            println!("conv_transpose2d: {d:?}");
        }
        Err(e) => println!("conv_transpose2d: ERR {e:?}"),
    }

    // gradient-penalty (exercises cg dweight/dbias): x[1,2,2,2], w[2,2,3,3], b[2].
    // torch: pen=6.025984 gw_sum=3.573120 gb=[1.93280, 1.08800].
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s
        .tensor_variable((0..8).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect(), vec![1, 2, 2, 2], true)
        .unwrap();
    let w = s
        .tensor_variable((0..36).map(|i| (i % 5) as f64 * 0.2 - 0.4).collect(), vec![2, 2, 3, 3], true)
        .unwrap();
    let b = s.tensor_variable(vec![0.1, -0.2], vec![2], true).unwrap();
    let y = s.functional_conv_transpose2d(x, w, Some(b), (1, 1), (0, 0), (0, 0)).unwrap();
    let sq = s.tensor_mul(y, y).unwrap();
    let loss = s.tensor_sum(sq).unwrap();
    let o = BackwardOptions::for_mode(s.mode()).with_create_graph(true).with_retain_graph(true);
    let r1 = s.tensor_backward_with_options(loss, o).unwrap();
    let g = r1.gradient_node(x).unwrap();
    let g2 = s.tensor_mul(g, g).unwrap();
    let pen = s.tensor_sum(g2).unwrap();
    let pv = s.tensor_values(pen).unwrap()[0];
    let gr = s.tensor_autograd_grad(&[pen], &[w, b], None, false, false).unwrap();
    let gw = gr[0].as_ref().unwrap();
    let gb = gr[1].as_ref().unwrap();
    println!("ct gp: pen={pv:.6} gw_sum={:.6} gb=[{:.5}, {:.5}]", gw.iter().sum::<f64>(), gb[0], gb[1]);
}
