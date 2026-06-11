//! In-place vs out-of-place parity probe. `tensor_<op>_` in-place variants are a
//! separate ad-hoc code path (closures in apply_tensor_unary_in_place) that does
//! NOT go through the typed kernel the out-of-place op uses, so they drift (stale
//! formulas, missing thresholds, typos). For each op this prints the max ULP/abs
//! difference between in_place(x) and out_of_place(x); anything nonzero is a bug
//! (the two MUST be identical). frankentorch in-place parity vein.
//!   cargo run -q -p ft-api --example inplace_parity_probe
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn main() {
    // broad domain incl extremes; unit-domain ops get a separate set
    let wide = vec![-800.0f64, -5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, 50.0, 800.0];
    let unit = vec![0.01f64, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99];
    let pos = vec![0.01f64, 0.5, 1.0, 2.0, 5.0, 20.0];

    macro_rules! cmp {
        ($name:literal, $oop:ident, $ip:ident, $inp:expr) => {{
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let a = s
                .tensor_variable($inp.clone(), vec![$inp.len()], false)
                .unwrap();
            let on = s.$oop(a).unwrap();
            let oref = s.tensor_values(on).unwrap();
            let b = s
                .tensor_variable($inp.clone(), vec![$inp.len()], false)
                .unwrap();
            s.$ip(b).unwrap();
            let ip = s.tensor_values(b).unwrap();
            let mut worst = 0.0f64;
            let mut wx = 0.0f64;
            let mut nan = false;
            for i in 0..$inp.len() {
                if ip[i].is_nan() != oref[i].is_nan() {
                    nan = true;
                }
                let d = (ip[i] - oref[i]).abs();
                if d > worst {
                    worst = d;
                    wx = $inp[i];
                }
            }
            let bitmiss = (0..$inp.len())
                .filter(|&i| ip[i].to_bits() != oref[i].to_bits())
                .count();
            let flag = if nan || worst > 0.0 {
                "  <-- DIVERGE"
            } else {
                ""
            };
            println!(
                "{:<13} bitmiss={}/{} maxabs={:.3e} @x={} nan_mismatch={}{}",
                $name,
                bitmiss,
                $inp.len(),
                worst,
                wx,
                nan,
                flag
            );
        }};
    }

    cmp!("silu", tensor_silu, tensor_silu_, wide);
    cmp!("mish", tensor_mish, tensor_mish_, wide);
    cmp!("softplus", tensor_softplus, tensor_softplus_, wide);
    cmp!("logsigmoid", tensor_logsigmoid, tensor_logsigmoid_, wide);
    cmp!("gelu", tensor_gelu, tensor_gelu_, wide);
    cmp!("erf", tensor_erf, tensor_erf_, wide);
    cmp!("erfc", tensor_erfc, tensor_erfc_, wide);
    cmp!("exp2", tensor_exp2, tensor_exp2_, wide);
    cmp!("cbrt", tensor_cbrt, tensor_cbrt_, wide);
    cmp!("tanhshrink", tensor_tanhshrink, tensor_tanhshrink_, wide);
    cmp!("softsign", tensor_softsign, tensor_softsign_, wide);
    cmp!("hardsigmoid", tensor_hardsigmoid, tensor_hardsigmoid_, wide);
    cmp!("hardswish", tensor_hardswish, tensor_hardswish_, wide);
    cmp!("sinc", tensor_sinc, tensor_sinc_, wide);
    cmp!("expit", tensor_sigmoid, tensor_expit_, wide);
    // positive-domain
    cmp!("lgamma", tensor_lgamma, tensor_lgamma_, pos);
    cmp!("digamma", tensor_digamma, tensor_digamma_, pos);
    cmp!("i0", tensor_i0, tensor_i0_, pos);
    cmp!("i1", tensor_i1, tensor_i1_, pos);
    cmp!("erfcx", tensor_erfcx, tensor_erfcx_, wide);
    // unit-domain (0,1)
    cmp!("erfinv", tensor_erfinv, tensor_erfinv_, unit);
    cmp!("ndtri", tensor_ndtri, tensor_ndtri_, unit);
    cmp!("ndtr", tensor_ndtr, tensor_ndtr_, wide);
}
