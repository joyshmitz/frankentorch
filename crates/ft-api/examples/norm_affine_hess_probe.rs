//! Does layer_norm/group_norm 2nd-order ERR WITH affine params? (the fused
//! borrowed-inputs grad path has no create_graph backward). frankentorch follow-up.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
fn main() {
    // layer_norm WITH weight+bias, [2,4] over last dim
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let xv: Vec<f64> = (0..8).map(|i| (i % 5) as f64 * 0.3 - 0.5).collect();
        let x = s.tensor_variable(xv, vec![2, 4], true).unwrap();
        let w = s
            .tensor_variable(vec![1.1, 0.9, 1.0, 1.2], vec![4], false)
            .unwrap();
        let b = s
            .tensor_variable(vec![0.1, -0.1, 0.0, 0.2], vec![4], false)
            .unwrap();
        let r = (|| {
            let y = s.functional_layer_norm(x, vec![4], Some(w), Some(b), 1e-5)?;
            let sq = s.tensor_mul(y, y)?;
            let l = s.tensor_sum(sq)?;
            s.tensor_functional_hessian(l, x)
        })();
        match r {
            Ok(h) => {
                let d: Vec<f64> = (0..8).map(|i| (h[i * 8 + i] * 1e4).round() / 1e4).collect();
                println!("layer_norm_affine: {d:?}");
            }
            Err(e) => println!("layer_norm_affine: ERR {e:?}"),
        }
    }
    // group_norm WITH weight+bias, [1,4,2,2] 2 groups
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let xv: Vec<f64> = (0..16).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect();
        let x = s.tensor_variable(xv, vec![1, 4, 2, 2], true).unwrap();
        let w = s
            .tensor_variable(vec![1.1, 0.9, 1.0, 1.2], vec![4], false)
            .unwrap();
        let b = s
            .tensor_variable(vec![0.1, -0.1, 0.0, 0.2], vec![4], false)
            .unwrap();
        let r = (|| {
            let y = s.functional_group_norm(x, 2, Some(w), Some(b), 1e-5)?;
            let sq = s.tensor_mul(y, y)?;
            let l = s.tensor_sum(sq)?;
            s.tensor_functional_hessian(l, x)
        })();
        match r {
            Ok(h) => {
                let d: Vec<f64> = (0..8)
                    .map(|i| (h[i * 16 + i] * 1e4).round() / 1e4)
                    .collect();
                println!("group_norm_affine: {d:?}");
            }
            Err(e) => println!("group_norm_affine: ERR {e:?}"),
        }
    }
}
