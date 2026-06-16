//! Double-backward (Hessian) probe for conv2d vs torch — the last untested
//! double-backward axis (matters for gradient-penalty / WGAN-GP training).
//! loss = sum(conv2d(x,w)^2) — quadratic in x (conv is linear in x), so the
//! Hessian over x is non-zero and constant. Hessian over the flattened input.
//!   cargo run -q -p ft-api --example conv2d_hessian_probe
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn main() {
    // x: [1,1,4,4] (16 elems), w: [1,1,3,3], valid conv -> [1,1,2,2].
    let xv: Vec<f64> = (0..16).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect();
    let wv: Vec<f64> = (0..9).map(|i| (i % 5) as f64 * 0.2 - 0.4).collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable(xv, vec![1, 1, 4, 4], true).unwrap();
    let w = s.tensor_variable(wv, vec![1, 1, 3, 3], false).unwrap();
    let build = |s: &mut FrankenTorchSession, x| -> Result<_, ft_autograd::AutogradError> {
        let y = s.functional_conv2d(x, w, None, (1, 1), (0, 0))?;
        let sq = s.tensor_mul(y, y)?;
        s.tensor_sum(sq)
    };
    match build(&mut s, x).and_then(|loss| s.tensor_functional_hessian(loss, x)) {
        Ok(h) => {
            let diag: Vec<f64> = (0..16).map(|i| (h[i * 16 + i] * 1e6).round() / 1e6).collect();
            println!("conv2d_sq diag={diag:?}");
            println!("conv2d_sq H[0,5]={:.6} H[0,10]={:.6}", h[5], h[10]);
        }
        Err(e) => println!("conv2d_sq ERR {e:?}"),
    }
}
