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

    let unit = vec![0.1f64, 0.3, 0.5, 0.7];
    let genv = vec![-1.0f64, 0.5, 1.0, 2.0];

    macro_rules! probe {
        ($name:literal, $m:ident) => {
            probe!($name, $m, x0)
        };
        ($name:literal, $m:ident, $inp:expr) => {{
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable($inp.clone(), vec![4], true).unwrap();
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
    // --- extended batch (frankentorch double-backward vein) ---
    probe!("log2", tensor_log2);
    probe!("log10", tensor_log10);
    probe!("exp2", tensor_exp2);
    probe!("rsqrt", tensor_rsqrt);
    probe!("acosh", tensor_acosh);
    probe!("lgamma", tensor_lgamma);
    probe!("digamma", tensor_digamma);
    probe!("asin", tensor_asin, unit);
    probe!("acos", tensor_acos, unit);
    probe!("atanh", tensor_atanh, unit);
    probe!("erfinv", tensor_erfinv, unit);
    probe!("i0", tensor_i0);
    probe!("i0e", tensor_i0e);
    probe!("i1", tensor_i1);
    probe!("i1e", tensor_i1e);
    probe!("erfcx", tensor_erfcx);
    probe!("ndtr", tensor_ndtr);
    probe!("ndtri", tensor_ndtri, unit);
    probe!("entr", tensor_entr);
    probe!("sinc", tensor_sinc);
    probe!("gelu_tanh", tensor_gelu_tanh, genv);
    probe!("softsign", tensor_softsign, genv);
    probe!("tanhshrink", tensor_tanhshrink, genv);
    probe!("selu", tensor_selu, genv);
    probe!("logsigmoid", tensor_logsigmoid, genv);
    let neg = vec![-2.0f64, -1.0, -0.5, -0.1];
    probe!("elu_neg", tensor_elu, neg);
    probe!("selu_neg", tensor_selu, neg);

    // --- composite / multi-arg ops (2nd derivative via create_graph) ---
    // torch goldens: pow3 [3,6,9,12]; recip [16,2,.5926,.25]; softmax [0,0,0,0];
    // logsoftmax [-.3649,-.5575,-.7993,-.9919]; prod [0,0,0,0]; sumsq [2,2,2,2].
    let mut cp =
        |name: &str,
         build: &dyn Fn(
            &mut FrankenTorchSession,
            ft_autograd::TensorNodeId,
        )
            -> Result<ft_autograd::TensorNodeId, ft_autograd::AutogradError>| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(x0.clone(), vec![4], true).unwrap();
            match build(&mut s, x)
                .and_then(|y| s.tensor_sum(y))
                .and_then(|out| s.tensor_functional_hessian(out, x))
            {
                Ok(h) => {
                    let d: Vec<f64> = (0..4).map(|i| (h[i * 4 + i] * 1e6).round() / 1e6).collect();
                    println!("{:<11} {:?}", name, d);
                }
                Err(e) => println!("{:<11} ERR {:?}", name, e),
            }
        };
    cp("pow3", &|s, x| s.tensor_pow(x, 3.0));
    cp("recip2", &|s, x| s.tensor_reciprocal(x));
    cp("softmax", &|s, x| s.tensor_softmax(x, 0));
    cp("logsoftmax", &|s, x| s.tensor_log_softmax(x, 0));
    cp("prod", &|s, x| s.tensor_prod(x));
    cp("sumsq", &|s, x| s.tensor_mul(x, x));
}
