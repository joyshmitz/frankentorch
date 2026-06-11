//! Reduction/statistical edge-case parity probe vs torch 2.12. Prints ft values;
//! compare to the torch oracle. Mismatch = a parity bug.
//!   cargo run -q -p ft-api --example reduction_edge_probe
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    macro_rules! p {
        ($name:literal, $e:expr) => {{
            match (|| -> Result<Vec<f64>, ft_autograd::AutogradError> { $e })() {
                Ok(v) => println!(
                    "{:<20} {:?}",
                    $name,
                    v.iter()
                        .map(|x| (x * 1e6).round() / 1e6)
                        .collect::<Vec<_>>()
                ),
                Err(e) => println!("{:<20} ERR {:?}", $name, e),
            }
        }};
    }
    let x = |s: &mut FrankenTorchSession| {
        s.tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], false)
            .unwrap()
    };
    let nanx = |s: &mut FrankenTorchSession| {
        s.tensor_variable(vec![1.0, f64::NAN, 3.0], vec![3], false)
            .unwrap()
    };

    // torch goldens (from the oracle run):
    // var_default 1.666667; var_unbiased_false 1.25; std_default 1.290994;
    // nanmean 2.0; nansum 4.0; nanmedian 1.0; median 2.0; q0.5 2.5; q0.25 1.75;
    // prod_empty 1.0; sum_empty 0.0; logsumexp 4.44019; std_corr2 1.581139
    p!("var_default(c=1)", {
        let t = x(&mut s);
        let r = s.tensor_var(t, 1)?;
        s.tensor_values(r)
    });
    p!("var_c=0", {
        let t = x(&mut s);
        let r = s.tensor_var(t, 0)?;
        s.tensor_values(r)
    });
    p!("std_default(c=1)", {
        let t = x(&mut s);
        let r = s.tensor_std(t, 1)?;
        s.tensor_values(r)
    });
    p!("std_c=2", {
        let t = x(&mut s);
        let r = s.tensor_std(t, 2)?;
        s.tensor_values(r)
    });
    p!("nanmean", {
        let t = nanx(&mut s);
        let r = s.tensor_nanmean(t)?;
        s.tensor_values(r)
    });
    p!("nanmedian", {
        let t = nanx(&mut s);
        let r = s.tensor_nanmedian(t)?;
        s.tensor_values(r)
    });
    p!("median_even", {
        let t = x(&mut s);
        let r = s.tensor_median(t)?;
        s.tensor_values(r)
    });
    p!("quantile_0.5", {
        let t = x(&mut s);
        let r = s.tensor_quantile(t, 0.5)?;
        s.tensor_values(r)
    });
    p!("quantile_0.25", {
        let t = x(&mut s);
        let r = s.tensor_quantile(t, 0.25)?;
        s.tensor_values(r)
    });
    p!("logsumexp", {
        let t = x(&mut s);
        let r = s.tensor_logsumexp(t, 0)?;
        s.tensor_values(r)
    });
    p!("prod_empty", {
        let t = s.tensor_variable(vec![], vec![0], false).unwrap();
        let r = s.tensor_prod(t)?;
        s.tensor_values(r)
    });
    p!("sum_empty", {
        let t = s.tensor_variable(vec![], vec![0], false).unwrap();
        let r = s.tensor_sum(t)?;
        s.tensor_values(r)
    });
}
