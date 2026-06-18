//! Finite-difference gradcheck for the VALUES of broadcasted binary-op gradients
//! (add/mul/sub/div with [3,1] op [1,4]) — catches wrong-axis sum-reduction in
//! the broadcast backward that shape checks alone would miss.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use ft_autograd::TensorNodeId;

const LA: [usize; 2] = [3, 1];
const LB: [usize; 2] = [1, 4];

fn apply(s: &mut FrankenTorchSession, name: &str, a: TensorNodeId, b: TensorNodeId) -> TensorNodeId {
    let o = match name {
        "add" => s.tensor_add(a, b),
        "mul" => s.tensor_mul(a, b),
        "sub" => s.tensor_sub(a, b),
        "div" => s.tensor_div(a, b),
        _ => unreachable!(),
    }
    .unwrap();
    s.tensor_sum(o).unwrap()
}

fn loss_at(name: &str, av: &[f64], bv: &[f64]) -> f64 {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let a = s.tensor_variable(av.to_vec(), LA.to_vec(), false).unwrap();
    let b = s.tensor_variable(bv.to_vec(), LB.to_vec(), false).unwrap();
    let l = apply(&mut s, name, a, b);
    s.tensor_values(l).unwrap()[0]
}

fn num_grad(name: &str, base: &[f64], other: &[f64], which: u8) -> Vec<f64> {
    let h = 1e-6;
    (0..base.len())
        .map(|i| {
            let (mut p, mut m) = (base.to_vec(), base.to_vec());
            p[i] += h;
            m[i] -= h;
            let (fp, fm) = if which == 0 {
                (loss_at(name, &p, other), loss_at(name, &m, other))
            } else {
                (loss_at(name, other, &p), loss_at(name, other, &m))
            };
            (fp - fm) / (2.0 * h)
        })
        .collect()
}

fn main() {
    let av = vec![0.5, 1.0, 1.5];
    let bv = vec![0.4, 0.7, 1.1, 1.6];
    for name in ["add", "mul", "sub", "div"] {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(av.clone(), LA.to_vec(), true).unwrap();
        let b = s.tensor_variable(bv.clone(), LB.to_vec(), true).unwrap();
        let l = apply(&mut s, name, a, b);
        let rep = s.tensor_backward(l).unwrap();
        let ga = s.tensor_gradient(&rep, a).unwrap().to_vec();
        let gb = s.tensor_gradient(&rep, b).unwrap().to_vec();
        let na = num_grad(name, &av, &bv, 0);
        let nb = num_grad(name, &bv, &av, 1);
        let rel = |a: &[f64], n: &[f64]| {
            a.iter()
                .zip(n)
                .map(|(x, y)| (x - y).abs() / (1.0 + y.abs()))
                .fold(0.0f64, f64::max)
        };
        let (ra, rb) = (rel(&ga, &na), rel(&gb, &nb));
        let verdict = if ra.max(rb) < 1e-4 { "PASS" } else { "FAIL" };
        println!("{name:<4} grad_a_relerr={ra:.3e} grad_b_relerr={rb:.3e} {verdict}");
        println!("     analytic_a={ga:?} numeric_a={na:?}");
        println!("     analytic_b={gb:?} numeric_b={nb:?}");
    }
}
