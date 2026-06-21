//! Structural-op double-backward (input-Hessian) probe vs torch — hunts for
//! silent-zero / ERR 2nd derivatives in the fused norm/pool custom ops, in both
//! f64 and f32. loss = sum(f(x)^2); prints Hessian diag (rounded) or ERR.
//!   cargo run -q --release -p ft-api --example struct_hessian_probe
use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, TensorNodeId};
use ft_core::{DType, ExecutionMode};

fn run(
    name: &str,
    xv: Vec<f64>,
    xs: Vec<usize>,
    f32: bool,
    f: impl Fn(&mut FrankenTorchSession, TensorNodeId) -> Result<TensorNodeId, AutogradError>,
    k: usize,
) {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let n: usize = xs.iter().product();
    match (|| {
        let mut x = s.tensor_variable(xv, xs, true)?;
        if f32 {
            x = s.tensor_to_dtype(x, DType::F32)?;
        }
        let y = f(&mut s, x)?;
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
                "{name:<22} diag={d:?}{}",
                if allzero { " (ALL ZERO)" } else { "" }
            );
        }
        Err(e) => println!("{name:<22} ERR {e:?}"),
    }
}

fn main() {
    let x16: Vec<f64> = (0..16).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect();
    for &f32 in &[false, true] {
        let tag = if f32 { "_f32" } else { "_f64" };
        // layer_norm over last dim (4), input [2,4]
        run(
            &format!("layer_norm{tag}"),
            (0..8).map(|i| (i % 5) as f64 * 0.3 - 0.5).collect(),
            vec![2, 4],
            f32,
            |s, x| s.functional_layer_norm(x, vec![4], None, None, 1e-5),
            8,
        );
        // rms_norm over last dim (4), input [2,4]
        run(
            &format!("rms_norm{tag}"),
            (0..8).map(|i| (i % 5) as f64 * 0.3 - 0.5).collect(),
            vec![2, 4],
            f32,
            |s, x| s.functional_rms_norm(x, vec![4], None, 1e-5),
            8,
        );
        // group_norm 2 groups, input [1,4,2,2]
        run(
            &format!("group_norm{tag}"),
            x16.clone(),
            vec![1, 4, 2, 2],
            f32,
            |s, x| s.functional_group_norm(x, 2, None, None, 1e-5),
            8,
        );
        // avg_pool2d 2x2 s2, input [1,1,4,4]
        run(
            &format!("avg_pool2d{tag}"),
            x16.clone(),
            vec![1, 1, 4, 4],
            f32,
            |s, x| s.functional_avg_pool2d(x, (2, 2), (2, 2), (0, 0), false, true),
            8,
        );
        // max_pool2d 2x2 s2, input [1,1,4,4]
        run(
            &format!("max_pool2d{tag}"),
            x16.clone(),
            vec![1, 1, 4, 4],
            f32,
            |s, x| s.functional_max_pool2d(x, (2, 2), (2, 2)),
            8,
        );
    }
}
