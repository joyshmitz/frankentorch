//! Gradient-penalty double-backward probe for functional_linear vs torch
//! (frankentorch-j4uio sibling): loss=sum(linear(x,w,b)^2); g=grad(loss,x,
//! create_graph); pen=sum(g^2); grad(pen,[w,b]). Exercises the cg dweight+dbias.
//!   cargo run -q --release -p ft-api --example linear_gradpenalty_probe
use ft_api::FrankenTorchSession;
use ft_autograd::BackwardOptions;
use ft_core::ExecutionMode;

fn main() {
    let xv: Vec<f64> = (0..3 * 4).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect();
    let wv: Vec<f64> = (0..2 * 4).map(|i| (i % 5) as f64 * 0.2 - 0.4).collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable(xv, vec![3, 4], true).unwrap();
    let w = s.tensor_variable(wv, vec![2, 4], true).unwrap();
    let b = s.tensor_variable(vec![0.1, -0.2], vec![2], true).unwrap();
    let y = s.functional_linear(x, w, Some(b)).unwrap();
    let sq = s.tensor_mul(y, y).unwrap();
    let loss = s.tensor_sum(sq).unwrap();
    let opts = BackwardOptions::for_mode(s.mode()).with_create_graph(true).with_retain_graph(true);
    let r1 = s.tensor_backward_with_options(loss, opts).unwrap();
    let g = r1.gradient_node(x).unwrap();
    let g2 = s.tensor_mul(g, g).unwrap();
    let pen = s.tensor_sum(g2).unwrap();
    let pv = s.tensor_values(pen).unwrap()[0];
    let gr = s.tensor_autograd_grad(&[pen], &[w, b], None, false, false).unwrap();
    let gw = gr[0].as_ref().unwrap();
    let gb = gr[1].as_ref().unwrap();
    let gw_sum: f64 = gw.iter().sum();
    println!("pen={pv:.6} gw_sum={gw_sum:.6} gw[0,0]={:.6} gw[1,3]={:.6} gb=[{:.4}, {:.4}]", gw[0], gw[7], gb[0], gb[1]);
}
