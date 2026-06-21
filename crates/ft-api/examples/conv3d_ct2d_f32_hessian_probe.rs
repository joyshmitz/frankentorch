//! conv3d-f32 and conv_transpose2d-f32 double-backward status probe vs torch f32
//! (frankentorch-lboou). Prints input-Hessian diag, ZERO, or ERR.
//!   cargo run -q --release -p ft-api --example conv3d_ct2d_f32_hessian_probe
use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, TensorNodeId};
use ft_core::{DType, ExecutionMode};

fn run(
    name: &str,
    xv: Vec<f64>,
    xs: Vec<usize>,
    wv: Vec<f64>,
    ws: Vec<usize>,
    f: impl Fn(
        &mut FrankenTorchSession,
        TensorNodeId,
        TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError>,
    k: usize,
) {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let n: usize = xs.iter().product();
    match (|| {
        let x = s.tensor_variable(xv, xs, true)?;
        let x = s.tensor_to_dtype(x, DType::F32)?;
        let w = s.tensor_variable(wv, ws, false)?;
        let w = s.tensor_to_dtype(w, DType::F32)?;
        let y = f(&mut s, x, w)?;
        let sq = s.tensor_mul(y, y)?;
        let loss = s.tensor_sum(sq)?;
        s.tensor_functional_hessian(loss, x)
    })() {
        Ok(h) => {
            let d: Vec<f64> = (0..k.min(n))
                .map(|i| (h[i * n + i] * 1e4).round() / 1e4)
                .collect();
            let allzero = h.iter().all(|&v| v == 0.0);
            println!(
                "{name}: diag={d:?}{}",
                if allzero { " (ALL ZERO)" } else { "" }
            );
        }
        Err(e) => println!("{name}: ERR {e:?}"),
    }
}

fn main() {
    // conv3d f32: N=1,C=1, 1x3x3 spatial, 1x1x2x2 kernel
    run(
        "conv3d_f32",
        (0..9).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect(),
        vec![1, 1, 1, 3, 3],
        (0..4).map(|i| (i % 5) as f64 * 0.2 - 0.4).collect(),
        vec![1, 1, 1, 2, 2],
        |s, x, w| s.functional_conv3d(x, w, None, (1, 1, 1), (0, 0, 0)),
        9,
    );
    // conv_transpose2d f32: N=1,C=1, 2x2 spatial, 1x1x2x2 kernel
    run(
        "conv_transpose2d_f32",
        vec![0.1, 0.2, -0.1, 0.3],
        vec![1, 1, 2, 2],
        vec![0.2, -0.4, 0.1, 0.3],
        vec![1, 1, 2, 2],
        |s, x, w| s.functional_conv_transpose2d(x, w, None, (1, 1), (0, 0), (0, 0)),
        4,
    );
}
