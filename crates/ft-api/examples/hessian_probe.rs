//! Double-backward (second-derivative) parity probe vs torch.
//! For s = sum(f(x)), the Hessian is diagonal with H[i,i] = f''(x_i).
//! Prints ft's Hessian diagonal for each elementwise op; diff against the
//! torch.autograd.functional.hessian oracle. A mismatch (esp. ft=0 where torch
//! is nonzero) means the op lacks a correct create_graph (differentiable) backward.
//!   cargo run -q -p ft-api --example hessian_probe
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn main() {
    let x0 = vec![0.5f64, 1.0, 1.5, 2.0];

    macro_rules! probe {
        ($name:literal, $m:ident) => {{
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(x0.clone(), vec![4], true).unwrap();
            match s.$m(x) {
                Ok(y) => match s.tensor_sum(y) {
                    Ok(out) => match s.tensor_functional_hessian(out, x) {
                        Ok(h) => {
                            let d: Vec<f64> =
                                (0..4).map(|i| (h[i * 4 + i] * 1e6).round() / 1e6).collect();
                            println!("{:<11} {:?}", $name, d);
                        }
                        Err(e) => println!("{:<11} HESS_ERR {:?}", $name, e),
                    },
                    Err(e) => println!("{:<11} SUM_ERR {:?}", $name, e),
                },
                Err(e) => println!("{:<11} OP_ERR {:?}", $name, e),
            }
        }};
    }

    probe!("sin", tensor_sin);
    probe!("cos", tensor_cos);
    probe!("exp", tensor_exp);
    probe!("log", tensor_log);
    probe!("tanh", tensor_tanh);
    probe!("sigmoid", tensor_sigmoid);
    probe!("sqrt", tensor_sqrt);
    probe!("reciprocal", tensor_reciprocal);
    probe!("softplus", tensor_softplus);
    probe!("gelu", tensor_gelu);
    probe!("silu", tensor_silu);
    probe!("erf", tensor_erf);
    probe!("asinh", tensor_asinh);
    probe!("expm1", tensor_expm1);
    probe!("log1p", tensor_log1p);
    probe!("sinh", tensor_sinh);
    probe!("cosh", tensor_cosh);
    probe!("atan", tensor_atan);
    probe!("tan", tensor_tan);
    probe!("elu", tensor_elu);
    probe!("mish", tensor_mish);
}
