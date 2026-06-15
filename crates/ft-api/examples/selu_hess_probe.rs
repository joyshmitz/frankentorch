//! Focused SELU/ELU second-derivative probe vs torch.
//! SELU/ELU are linear for x>0 so the Hessian diagonal must be 0 there.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn hess(s: &mut FrankenTorchSession, name: &str, inp: Vec<f64>, op: &str) {
    let x = s
        .tensor_variable(inp.clone(), vec![inp.len()], true)
        .unwrap();
    let y = match op {
        "selu" => s.tensor_selu(x),
        "elu" => s.tensor_elu(x),
        _ => unreachable!(),
    }
    .unwrap();
    let out = s.tensor_sum(y).unwrap();
    let h = s.tensor_functional_hessian(out, x).unwrap();
    let n = inp.len();
    let d: Vec<f64> = (0..n).map(|i| (h[i * n + i] * 1e6).round() / 1e6).collect();
    println!("{name:<14} {d:?}");
}

fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    hess(&mut s, "selu pos", vec![0.5, 1.0, 1.5, 2.0], "selu");
    hess(&mut s, "selu mix", vec![-1.0, 0.5, 1.0, 2.0], "selu");
    hess(&mut s, "selu single", vec![0.5], "selu");
    hess(&mut s, "selu single2", vec![1.5], "selu");
    hess(&mut s, "elu pos", vec![0.5, 1.0, 1.5, 2.0], "elu");
    hess(&mut s, "elu mix", vec![-1.0, 0.5, 1.0, 2.0], "elu");
}
