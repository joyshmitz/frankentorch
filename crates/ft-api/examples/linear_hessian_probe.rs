//! Double-backward (Hessian) probe for functional_linear vs torch — the fused
//! f64 linear grad fast path (linear_tensor_f64 / linear_backward_f64) had no
//! create_graph backward, so 2nd-order (Hessian / gradient-penalty over linear
//! layers — the dominant MLP/transformer op) ERR'd. frankentorch-j4uio sibling.
//! loss = sum(linear(x,w)^2); Hessian over the flattened input.
//!   cargo run -q --release -p ft-api --example linear_hessian_probe
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn main() {
    let xv: Vec<f64> = (0..3 * 4).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect();
    let wv: Vec<f64> = (0..2 * 4).map(|i| (i % 5) as f64 * 0.2 - 0.4).collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable(xv, vec![3, 4], true).unwrap();
    let w = s.tensor_variable(wv, vec![2, 4], false).unwrap();
    let build = |s: &mut FrankenTorchSession, x| -> Result<_, ft_autograd::AutogradError> {
        let y = s.functional_linear(x, w, None)?;
        let sq = s.tensor_mul(y, y)?;
        s.tensor_sum(sq)
    };
    match build(&mut s, x).and_then(|loss| s.tensor_functional_hessian(loss, x)) {
        Ok(h) => {
            let diag: Vec<f64> = (0..12)
                .map(|i| (h[i * 12 + i] * 1e6).round() / 1e6)
                .collect();
            println!("linear_sq diag={diag:?}");
            println!("linear_sq H[0,4]={:.6} H[0,8]={:.6}", h[4], h[8]);
        }
        Err(e) => println!("linear_sq ERR {e:?}"),
    }
}
