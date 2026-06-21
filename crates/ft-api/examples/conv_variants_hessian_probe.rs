//! Double-backward status probe for conv VARIANTS vs torch (frankentorch-lboou).
//! Prints the input-Hessian diagonal (or ERR) for conv1d / conv2d_dilated /
//! conv2d_grouped / conv3d so we know which still need the j4uio cg recipe and
//! which already work for free (compose over the now-fixed functional_conv2d).
//!   cargo run -q --release -p ft-api --example conv_variants_hessian_probe
use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, TensorNodeId};
use ft_core::ExecutionMode;

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
    show: usize,
) {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let n: usize = xs.iter().product();
    let x = s.tensor_variable(xv, xs, true).unwrap();
    let w = s.tensor_variable(wv, ws, false).unwrap();
    let built = (|| {
        let y = f(&mut s, x, w)?;
        let sq = s.tensor_mul(y, y)?;
        let loss = s.tensor_sum(sq)?;
        s.tensor_functional_hessian(loss, x)
    })();
    match built {
        Ok(h) => {
            let d: Vec<f64> = (0..show.min(n))
                .map(|i| (h[i * n + i] * 1e5).round() / 1e5)
                .collect();
            println!("{name}: {d:?}");
        }
        Err(e) => println!("{name}: ERR {e:?}"),
    }
}

fn main() {
    run(
        "conv1d",
        (0..6).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect(),
        vec![1, 1, 6],
        vec![0.2, -0.4, 0.1],
        vec![1, 1, 3],
        |s, x, w| s.functional_conv1d(x, w, None, 1, 0),
        6,
    );
    run(
        "conv2d_dilated",
        (0..25).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect(),
        vec![1, 1, 5, 5],
        (0..9).map(|i| (i % 5) as f64 * 0.2 - 0.4).collect(),
        vec![1, 1, 3, 3],
        |s, x, w| s.functional_conv2d_dilated(x, w, None, (1, 1), (0, 0), (2, 2), 1),
        8,
    );
    run(
        "conv2d_grouped",
        (0..32).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect(),
        vec![1, 2, 4, 4],
        (0..18).map(|i| (i % 5) as f64 * 0.2 - 0.4).collect(),
        vec![2, 1, 3, 3],
        |s, x, w| s.functional_conv2d_grouped(x, w, None, (1, 1), (0, 0), 2),
        8,
    );
    run(
        "conv3d",
        (0..64).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect(),
        vec![1, 1, 4, 4, 4],
        (0..8).map(|i| (i % 5) as f64 * 0.2 - 0.4).collect(),
        vec![1, 1, 2, 2, 2],
        |s, x, w| s.functional_conv3d(x, w, None, (1, 1, 1), (0, 0, 0)),
        8,
    );
}
