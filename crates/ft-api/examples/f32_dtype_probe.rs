//! f32 dtype-parity probe (frankentorch project_f32_parity_vein). torch keeps
//! the input float dtype for elementwise + reduction float ops; ft sometimes
//! errors on f32 (tensor_values is f64-only) or silently upcasts to F64. Each
//! op below is run on an f32 input; anything that ERRs or returns dtype != F32
//! is a parity bug.
//!
//!   cargo run -q -p ft-api --example f32_dtype_probe

use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};

fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let mut bugs = 0u32;

    macro_rules! un {
        ($name:literal, $m:ident, $vals:expr) => {{
            let t = s.tensor_variable_f32($vals, vec![4], false).unwrap();
            match s.$m(t) {
                Ok(r) => match s.tensor_dtype(r) {
                    Ok(DType::F32) => {}
                    Ok(d) => {
                        println!("{:<28} UPCAST -> {:?}  (torch keeps F32)", $name, d);
                        bugs += 1;
                    }
                    Err(e) => println!("{:<28} dtype-err {e:?}", $name),
                },
                Err(e) => {
                    println!("{:<28} ERR {e:?}", $name);
                    bugs += 1;
                }
            }
        }};
    }

    // positive-domain inputs (valid for log/sqrt/bessel/gamma/etc.)
    let pos = || vec![0.5f32, 1.0, 2.0, 3.0];
    // general inputs
    let genv = || vec![-1.0f32, 0.5, 1.0, 2.0];
    // (0,1) inputs (for ndtri/logit/atanh/erfinv)
    let unit = || vec![0.1f32, 0.3, 0.5, 0.9];

    un!("digamma", tensor_digamma, pos());
    un!("i0", tensor_i0, pos());
    un!("i1", tensor_i1, pos());
    un!("i0e", tensor_i0e, pos());
    un!("i1e", tensor_i1e, pos());
    un!("erf", tensor_erf, genv());
    un!("erfc", tensor_erfc, genv());
    un!("erfcx", tensor_erfcx, genv());
    un!("erfinv", tensor_erfinv, vec![-0.5f32, 0.0, 0.5, 0.9]);
    un!("ndtr", tensor_ndtr, genv());
    un!("ndtri", tensor_ndtri, unit());
    un!("lgamma", tensor_lgamma, pos());
    un!("sinc", tensor_sinc, genv());
    un!("expm1", tensor_expm1, genv());
    un!("log1p", tensor_log1p, pos());
    un!("exp2", tensor_exp2, genv());
    un!("log2", tensor_log2, pos());
    un!("log10", tensor_log10, pos());
    un!("special_bessel_j0", tensor_special_bessel_j0, pos());
    un!("special_bessel_j1", tensor_special_bessel_j1, pos());
    un!("special_bessel_y0", tensor_special_bessel_y0, pos());
    un!("special_modified_bessel_k0", tensor_special_modified_bessel_k0, pos());
    un!("gelu", tensor_gelu, genv());
    un!("silu", tensor_silu, genv());
    un!("mish", tensor_mish, genv());
    un!("softplus", tensor_softplus, genv());
    un!("softsign", tensor_softsign, genv());
    un!("reciprocal", tensor_reciprocal, genv());
    un!("sign", tensor_sign, genv());
    un!("trunc", tensor_trunc, genv());
    un!("frac", tensor_frac, genv());
    un!("nansum", tensor_nansum, genv());
    un!("nanmean", tensor_nanmean, genv());
    un!("prod", tensor_prod, pos());

    println!("\nf32 UPCAST/ERR bugs: {bugs}");
}
