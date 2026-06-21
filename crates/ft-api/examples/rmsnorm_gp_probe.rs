//! rms_norm double-backward: with-affine input-Hessian + full gradient-penalty
//! (pen=sum(grad(sum(y^2),x,cg)^2); grad(pen,[x,w])) vs torch. frankentorch-9zo9m.
use ft_api::FrankenTorchSession;
use ft_autograd::BackwardOptions;
use ft_core::{DType, ExecutionMode};
fn main() {
    for f32 in [false, true] {
        let tag = if f32 { "f32" } else { "f64" };
        // with-weight input-Hessian
        {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let xv: Vec<f64> = (0..8).map(|i| (i % 5) as f64 * 0.3 - 0.5).collect();
            let mut x = s.tensor_variable(xv, vec![2, 4], true).unwrap();
            let mut w = s
                .tensor_variable(vec![1.1, 0.9, 1.0, 1.2], vec![4], false)
                .unwrap();
            if f32 {
                x = s.tensor_to_dtype(x, DType::F32).unwrap();
                w = s.tensor_to_dtype(w, DType::F32).unwrap();
            }
            let r = (|| {
                let y = s.functional_rms_norm(x, vec![4], Some(w), 1e-5)?;
                let sq = s.tensor_mul(y, y)?;
                let l = s.tensor_sum(sq)?;
                s.tensor_functional_hessian(l, x)
            })();
            match r {
                Ok(h) => {
                    let d: Vec<f64> = (0..8).map(|i| (h[i * 8 + i] * 1e4).round() / 1e4).collect();
                    println!("rms_w_hess_{tag}: {d:?}");
                }
                Err(e) => println!("rms_w_hess_{tag}: ERR {e:?}"),
            }
        }
        // full GP (grad wrt x and w)
        {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let xv: Vec<f64> = (0..8).map(|i| (i % 5) as f64 * 0.3 - 0.5).collect();
            let mut x = s.tensor_variable(xv, vec![2, 4], true).unwrap();
            let mut w = s
                .tensor_variable(vec![1.1, 0.9, 1.0, 1.2], vec![4], true)
                .unwrap();
            if f32 {
                x = s.tensor_to_dtype(x, DType::F32).unwrap();
                w = s.tensor_to_dtype(w, DType::F32).unwrap();
            }
            let r = (|| {
                let y = s.functional_rms_norm(x, vec![4], Some(w), 1e-5)?;
                let sq = s.tensor_mul(y, y)?;
                let l = s.tensor_sum(sq)?;
                let o = BackwardOptions::for_mode(s.mode())
                    .with_create_graph(true)
                    .with_retain_graph(true);
                let r1 = s.tensor_backward_with_options(l, o)?;
                let g = r1.gradient_node(x).unwrap();
                let g2 = s.tensor_mul(g, g)?;
                let pen = s.tensor_sum(g2)?;
                let pv = s.tensor_values(pen)?[0];
                let gr = s.tensor_autograd_grad(&[pen], &[x, w], None, false, false)?;
                Ok::<_, ft_autograd::AutogradError>((
                    pv,
                    gr[0].clone().unwrap().iter().sum::<f64>(),
                    gr[1].clone().unwrap().iter().sum::<f64>(),
                ))
            })();
            match r {
                Ok((p, gx, gw)) => {
                    println!("rms_gp_{tag}: pen={p:.4} gx_sum={gx:.4} gw_sum={gw:.4}")
                }
                Err(e) => println!("rms_gp_{tag}: ERR {e:?}"),
            }
        }
    }
}
