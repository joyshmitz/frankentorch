//! conv3d-f32 + conv_transpose2d-f32 full gradient-penalty VALUE check vs torch
//! f32 (frankentorch-lboou): pen=sum(grad(sum(y^2),x,create_graph)^2), then
//! grad(pen,[weight,bias]) — exercises the input/weight/bias cg branches.
use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, BackwardOptions, TensorNodeId};
use ft_core::{DType, ExecutionMode};

fn gp(
    name: &str,
    xv: Vec<f64>,
    xs: Vec<usize>,
    wv: Vec<f64>,
    ws: Vec<usize>,
    bv: Vec<f64>,
    f: impl Fn(&mut FrankenTorchSession, TensorNodeId, TensorNodeId, TensorNodeId) -> Result<TensorNodeId, AutogradError>,
) {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x0 = s.tensor_variable(xv, xs, true).unwrap();
    let x = s.tensor_to_dtype(x0, DType::F32).unwrap();
    let w0 = s.tensor_variable(wv, ws, true).unwrap();
    let w = s.tensor_to_dtype(w0, DType::F32).unwrap();
    let bl = bv.len();
    let b0 = s.tensor_variable(bv, vec![bl], true).unwrap();
    let b = s.tensor_to_dtype(b0, DType::F32).unwrap();
    let r = (|| {
        let y = f(&mut s, x, w, b)?;
        let sq = s.tensor_mul(y, y)?;
        let loss = s.tensor_sum(sq)?;
        let o = BackwardOptions::for_mode(s.mode())
            .with_create_graph(true)
            .with_retain_graph(true);
        let r1 = s.tensor_backward_with_options(loss, o)?;
        let g = r1.gradient_node(x).unwrap();
        let g2 = s.tensor_mul(g, g)?;
        let pen = s.tensor_sum(g2)?;
        let pv = s.tensor_values(pen)?[0];
        let gr = s.tensor_autograd_grad(&[pen], &[w, b], None, false, false)?;
        Ok::<_, AutogradError>((
            pv,
            gr[0].clone().unwrap().iter().sum::<f64>(),
            gr[1].clone().unwrap(),
        ))
    })();
    match r {
        Ok((p, gws, gb)) => {
            let g: Vec<f64> = gb.iter().map(|v| (v * 1e4).round() / 1e4).collect();
            println!("{name}: pen={p:.4} gw_sum={gws:.4} gb={g:?}");
        }
        Err(e) => println!("{name}: ERR {e:?}"),
    }
}

fn main() {
    gp(
        "conv3d_f32",
        (0..18).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect(),
        vec![1, 1, 2, 3, 3],
        (0..4).map(|i| (i % 5) as f64 * 0.2 - 0.4).collect(),
        vec![1, 1, 1, 2, 2],
        vec![0.1],
        |s, x, w, b| s.functional_conv3d(x, w, Some(b), (1, 1, 1), (0, 0, 0)),
    );
    gp(
        "conv_transpose2d_f32",
        vec![0.1, 0.2, -0.1, 0.3, 0.05, -0.15, 0.25, -0.05],
        vec![1, 2, 2, 2],
        (0..8).map(|i| (i % 5) as f64 * 0.2 - 0.4).collect(),
        vec![2, 1, 2, 2],
        vec![0.1],
        |s, x, w, b| s.functional_conv_transpose2d(x, w, Some(b), (1, 1), (0, 0), (0, 0)),
    );
}
