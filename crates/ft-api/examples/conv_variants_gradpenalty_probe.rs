//! Gradient-penalty (weight+bias double-backward) probe for conv3d + grouped
//! conv vs torch (frankentorch-lboou). The input-Hessian probe leaves weight
//! fixed, so it does NOT exercise the cg dweight/dbias branches of the new
//! kernels — this does. loss=sum(conv(x,w,b)^2); g=grad(loss,x,create_graph);
//! pen=sum(g^2); grad(pen,[w,b]).
//!   cargo run -q --release -p ft-api --example conv_variants_gradpenalty_probe
use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, BackwardOptions, TensorNodeId};
use ft_core::ExecutionMode;

#[allow(clippy::too_many_arguments)]
fn gp(
    name: &str,
    xv: Vec<f64>,
    xs: Vec<usize>,
    wv: Vec<f64>,
    ws: Vec<usize>,
    bv: Vec<f64>,
    bs: Vec<usize>,
    f: impl Fn(
        &mut FrankenTorchSession,
        TensorNodeId,
        TensorNodeId,
        TensorNodeId,
    ) -> Result<TensorNodeId, AutogradError>,
) {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable(xv, xs, true).unwrap();
    let w = s.tensor_variable(wv, ws, true).unwrap();
    let b = s.tensor_variable(bv, bs, true).unwrap();
    let r = (|| {
        let y = f(&mut s, x, w, b)?;
        let sq = s.tensor_mul(y, y)?;
        let loss = s.tensor_sum(sq)?;
        let opts = BackwardOptions::for_mode(s.mode())
            .with_create_graph(true)
            .with_retain_graph(true);
        let r1 = s.tensor_backward_with_options(loss, opts)?;
        let g = r1.gradient_node(x).unwrap();
        let g2 = s.tensor_mul(g, g)?;
        let pen = s.tensor_sum(g2)?;
        let pv = s.tensor_values(pen)?[0];
        let gr = s.tensor_autograd_grad(&[pen], &[w, b], None, false, false)?;
        let gw = gr[0].clone().unwrap();
        let gb = gr[1].clone().unwrap();
        Ok::<_, AutogradError>((pv, gw.iter().sum::<f64>(), gb))
    })();
    match r {
        Ok((pen, gw_sum, gb)) => {
            let gbr: Vec<f64> = gb.iter().map(|v| (v * 1e5).round() / 1e5).collect();
            println!("{name} gp: pen={pen:.6} gw_sum={gw_sum:.6} gb={gbr:?}");
        }
        Err(e) => println!("{name} gp: ERR {e:?}"),
    }
}

fn main() {
    gp(
        "conv3d",
        (0..64).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect(),
        vec![1, 1, 4, 4, 4],
        (0..16).map(|i| (i % 5) as f64 * 0.2 - 0.4).collect(),
        vec![2, 1, 2, 2, 2],
        vec![0.1, -0.2],
        vec![2],
        |s, x, w, b| s.functional_conv3d(x, w, Some(b), (1, 1, 1), (0, 0, 0)),
    );
    gp(
        "grouped",
        (0..32).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect(),
        vec![1, 2, 4, 4],
        (0..18).map(|i| (i % 5) as f64 * 0.2 - 0.4).collect(),
        vec![2, 1, 3, 3],
        vec![0.1, -0.2],
        vec![2],
        |s, x, w, b| s.functional_conv2d_grouped(x, w, Some(b), (1, 1), (0, 0), 2),
    );
}
