//! f16/bf16 dtype-parity probe (project_half_movement_vein). torch keeps the
//! input half dtype for elementwise + reduction compute ops; ft may error or
//! upcast (typed dispatch often has an f32 arm but no f16/bf16 arm). Each op is
//! run on an f16 input; anything that ERRs or returns dtype != F16 is a bug.
//!
//!   cargo run -q -p ft-api --example half_dtype_probe

use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};

fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let mut bugs = 0u32;

    // f16 input of shape [4].
    macro_rules! h4 {
        ($s:expr, $v:expr) => {{
            let f = $s.tensor_variable_f32($v, vec![4], false).unwrap();
            $s.tensor_half(f).unwrap()
        }};
    }
    macro_rules! un {
        ($name:literal, $m:ident, $v:expr) => {{
            let t = h4!(s, $v);
            match s.$m(t) {
                Ok(r) => match s.tensor_dtype(r) {
                    Ok(DType::F16) => {}
                    Ok(d) => { println!("{:<24} UPCAST -> {:?}", $name, d); bugs += 1; }
                    Err(e) => println!("{:<24} dtype-err {e:?}", $name),
                },
                Err(e) => { println!("{:<24} ERR {e:?}", $name); bugs += 1; }
            }
        }};
    }
    let p = || vec![0.5f32, 1.0, 2.0, 3.0];
    let g = || vec![-1.0f32, 0.5, 1.0, 2.0];
    un!("sin", tensor_sin, g());
    un!("cos", tensor_cos, g());
    un!("exp", tensor_exp, g());
    un!("log", tensor_log, p());
    un!("sqrt", tensor_sqrt, p());
    un!("rsqrt", tensor_rsqrt, p());
    un!("abs", tensor_abs, g());
    un!("neg", tensor_neg, g());
    un!("reciprocal", tensor_reciprocal, p());
    un!("sigmoid", tensor_sigmoid, g());
    un!("tanh", tensor_tanh, g());
    un!("gelu", tensor_gelu, g());
    un!("relu", tensor_relu, g());
    un!("erf", tensor_erf, g());
    un!("sum", tensor_sum, p());
    un!("mean", tensor_mean, p());

    // binary (both f16 -> f16)
    macro_rules! bin {
        ($name:literal, $call:expr) => {{
            let x = h4!(s, p());
            let y = h4!(s, p());
            match $call(&mut s, x, y) {
                Ok(r) => match s.tensor_dtype(r) {
                    Ok(DType::F16) => {}
                    Ok(d) => { println!("{:<24} UPCAST -> {:?}", $name, d); bugs += 1; }
                    Err(e) => println!("{:<24} dtype-err {e:?}", $name),
                },
                Err(e) => { println!("{:<24} ERR {e:?}", $name); bugs += 1; }
            }
        }};
    }
    bin!("add", |s: &mut FrankenTorchSession, x, y| s.tensor_add(x, y));
    bin!("mul", |s: &mut FrankenTorchSession, x, y| s.tensor_mul(x, y));
    bin!("div", |s: &mut FrankenTorchSession, x, y| s.tensor_div(x, y));
    bin!("sub", |s: &mut FrankenTorchSession, x, y| s.tensor_sub(x, y));
    bin!("matmul", |s: &mut FrankenTorchSession, x, y| s.tensor_matmul(x, y));

    // dim ops
    let t = h4!(s, p());
    match s.tensor_softmax(t, 0) {
        Ok(r) => { if s.tensor_dtype(r).unwrap() != DType::F16 { println!("{:<24} UPCAST -> {:?}", "softmax", s.tensor_dtype(r).unwrap()); bugs += 1; } }
        Err(e) => { println!("{:<24} ERR {e:?}", "softmax"); bugs += 1; }
    }
    let t = h4!(s, p());
    match s.tensor_pow(t, 2.0) {
        Ok(r) => { if s.tensor_dtype(r).unwrap() != DType::F16 { println!("{:<24} UPCAST -> {:?}", "pow", s.tensor_dtype(r).unwrap()); bugs += 1; } }
        Err(e) => { println!("{:<24} ERR {e:?}", "pow"); bugs += 1; }
    }

    println!("\nf16 UPCAST/ERR bugs: {bugs}");
}
