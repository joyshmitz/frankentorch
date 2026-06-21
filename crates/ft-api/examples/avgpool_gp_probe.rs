//! avg_pool2d double-backward parity: input-Hessian diag under padding + cip
//! variants, vs torch. frankentorch-cqmed.
use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
fn hd(name: &str, f32: bool, pad: (usize, usize), cip: bool) {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let xv: Vec<f64> = (0..16).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect();
    let mut x = s.tensor_variable(xv, vec![1, 1, 4, 4], true).unwrap();
    if f32 {
        x = s.tensor_to_dtype(x, DType::F32).unwrap();
    }
    let r = (|| {
        let y = s.functional_avg_pool2d(x, (2, 2), (2, 2), pad, false, cip)?;
        let sq = s.tensor_mul(y, y)?;
        let l = s.tensor_sum(sq)?;
        s.tensor_functional_hessian(l, x)
    })();
    match r {
        Ok(h) => {
            let d: Vec<f64> = (0..16)
                .map(|i| (h[i * 16 + i] * 1e4).round() / 1e4)
                .collect();
            println!("{name}: {d:?}");
        }
        Err(e) => println!("{name}: ERR {e:?}"),
    }
}
fn main() {
    hd("nopad_cip_f64", false, (0, 0), true);
    hd("nopad_cip_f32", true, (0, 0), true);
    hd("pad1_cip_f64", false, (1, 1), true);
    hd("pad1_cip_f32", true, (1, 1), true);
    hd("pad1_nocip_f64", false, (1, 1), false);
    hd("pad1_nocip_f32", true, (1, 1), false);
}
