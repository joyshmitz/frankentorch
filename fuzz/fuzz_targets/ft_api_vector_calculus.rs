#![no_main]

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 64;

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let a: Vec<f64> = data[..3]
        .iter()
        .map(|&b| (b as i32 - 128) as f64 / 20.0)
        .collect();
    let b_vec: Vec<f64> = data[3..6]
        .iter()
        .map(|&b| (b as i32 - 128) as f64 / 20.0)
        .collect();

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let av = match s.tensor_variable(a.clone(), vec![3], false) {
        Ok(t) => t,
        Err(_) => return,
    };
    let bv = match s.tensor_variable(b_vec.clone(), vec![3], false) {
        Ok(t) => t,
        Err(_) => return,
    };

    // cross(a, b)
    let cross_ab = match s.tensor_cross(av, bv) {
        Ok(t) => t,
        Err(_) => return,
    };
    let v_ab = s.tensor_values(cross_ab).expect("values");
    assert_eq!(v_ab.len(), 3, "cross output length");

    // cross(b, a) — antisymmetry: cross(a, b) == -cross(b, a)
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let av2 = s.tensor_variable(a.clone(), vec![3], false).expect("av2");
    let bv2 = s.tensor_variable(b_vec.clone(), vec![3], false).expect("bv2");
    let cross_ba = s.tensor_cross(bv2, av2).expect("cross_ba");
    let v_ba = s.tensor_values(cross_ba).expect("v_ba");

    let abs_max = a.iter().chain(b_vec.iter()).fold(0.0_f64, |acc, &v| acc.max(v.abs())).max(1.0);
    let bound = 32.0 * f64::EPSILON * abs_max * abs_max + 1e-12;
    for i in 0..3 {
        let sum = v_ab[i] + v_ba[i];
        assert!(
            sum.abs() <= bound,
            "cross antisymmetry broken at [{i}]: ab={} + ba={} = {} (bound={:e})",
            v_ab[i], v_ba[i], sum, bound
        );
    }

    // cross(a, a) is the zero vector.
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let aa1 = s.tensor_variable(a.clone(), vec![3], false).expect("aa1");
    let aa2 = s.tensor_variable(a.clone(), vec![3], false).expect("aa2");
    let cross_aa = s.tensor_cross(aa1, aa2).expect("cross_aa");
    let v_aa = s.tensor_values(cross_aa).expect("v_aa");
    for (i, &v) in v_aa.iter().enumerate() {
        assert!(
            v.abs() <= 32.0 * f64::EPSILON * abs_max * abs_max + 1e-15,
            "cross(a, a)[{i}] = {v} should be 0 (a = {:?})", a
        );
    }
});
