//! Backward-gradient parity probe at EXTREME inputs vs torch. For s = sum(f(x)),
//! d s/d x_i = f'(x_i). Backward formulas often overflow/NaN at large |x| or
//! near-singular inputs even when the forward is fine (cf. the softplus forward
//! overflow bug). Prints ft's grad at x ∈ {-1000,-50,-1,1,50,1000}; compare to the
//! torch oracle. ft=NaN/inf where torch is finite (or wrong value) = a bug.
//!   cargo run -q -p ft-api --example backward_extremes_probe
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn main() {
    let xs = vec![-1000.0f64, -50.0, -1.0, 1.0, 50.0, 1000.0];

    macro_rules! probe {
        ($name:literal, $m:ident) => {
            probe!($name, $m, xs)
        };
        ($name:literal, $m:ident, $inp:expr) => {{
            let n = $inp.len();
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable($inp.clone(), vec![n], true).unwrap();
            match s.$m(x) {
                Ok(y) => {
                    let out = s.tensor_sum(y).unwrap();
                    match s.tensor_backward(out) {
                        Ok(rep) => {
                            let g = s.tensor_gradient(&rep, x).unwrap();
                            let d: Vec<f64> = (0..n).map(|i| (g[i] * 1e6).round() / 1e6).collect();
                            let bad = g.iter().any(|v| !v.is_finite());
                            println!(
                                "{:<11} {:?}{}",
                                $name,
                                d,
                                if bad { "  <-- NONFINITE" } else { "" }
                            );
                        }
                        Err(e) => println!("{:<11} BWD_ERR {:?}", $name, e),
                    }
                }
                Err(e) => println!("{:<11} OP_ERR {:?}", $name, e),
            }
        }};
    }

    probe!("sigmoid", tensor_sigmoid);
    probe!("tanh", tensor_tanh);
    probe!("softplus", tensor_softplus);
    probe!("silu", tensor_silu);
    probe!("mish", tensor_mish);
    probe!("gelu", tensor_gelu);
    probe!("exp", tensor_exp);
    probe!("sinh", tensor_sinh);
    probe!("cosh", tensor_cosh);
    probe!("logsigmoid", tensor_logsigmoid);
    probe!("softsign", tensor_softsign);
    probe!("tanhshrink", tensor_tanhshrink);
    probe!("elu", tensor_elu);
    probe!("selu", tensor_selu);

    // --- singular-gradient inputs: torch gives inf at x=0 (sqrt/log/rsqrt/...) ---
    println!("-- singular inputs [0, 1e-300, 1e-12, 1e-6] (torch: inf at 0) --");
    let sing = vec![0.0f64, 1e-300, 1e-12, 1e-6];
    probe!("sqrt", tensor_sqrt, sing);
    probe!("rsqrt", tensor_rsqrt, sing);
    probe!("log", tensor_log, sing);
    probe!("reciprocal", tensor_reciprocal, sing);
    println!("-- acos/asin near +-1 (torch: inf at 1) --");
    let near1 = vec![1.0f64, 0.999999, -1.0];
    probe!("acos", tensor_acos, near1);
    probe!("asin", tensor_asin, near1);
}
