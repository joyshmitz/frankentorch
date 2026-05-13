#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{
    add_tensor_contiguous_f64, div_tensor_contiguous_f64, eq_tensor_contiguous_f64,
    lt_tensor_contiguous_f64, mul_tensor_contiguous_f64, neg_tensor_contiguous_f64,
    sub_tensor_contiguous_f64,
};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let ndim = usize::from(data[0] % 5);
    if ndim == 0 {
        return;
    }
    let body = &data[1..];

    if body.len() < ndim {
        return;
    }
    let shape: Vec<usize> = body[..ndim]
        .iter()
        .map(|b| usize::from(b % (MAX_SHAPE_DIM + 1)))
        .collect();

    let meta = match TensorMeta::from_shape_and_strides(
        shape.clone(),
        ft_core::contiguous_strides(&shape),
        0,
        DType::F64,
        Device::Cpu,
    ) {
        Ok(meta) => meta,
        Err(_) => return,
    };
    let numel = meta.numel();
    if numel > 4096 {
        return;
    }

    // Mix adversarial specials with finite values.
    let storage_for = |seed_offset: usize| -> Vec<f64> {
        (0..numel)
            .map(|i| {
                let raw = body[(seed_offset + i) % body.len().max(1)];
                match raw % 16 {
                    0 => f64::NAN,
                    1 => f64::INFINITY,
                    2 => f64::NEG_INFINITY,
                    _ => f64::from(raw as i8) / 20.0,
                }
            })
            .collect()
    };
    let a = storage_for(ndim);
    let b = storage_for(ndim + 7);

    // Length contract for each op.
    let add_ab = add_tensor_contiguous_f64(&a, &b, &meta, &meta).expect("add");
    let sub_ab = sub_tensor_contiguous_f64(&a, &b, &meta, &meta).expect("sub");
    let mul_ab = mul_tensor_contiguous_f64(&a, &b, &meta, &meta).expect("mul");
    let div_ab = div_tensor_contiguous_f64(&a, &b, &meta, &meta).expect("div");
    let eq_ab = eq_tensor_contiguous_f64(&a, &b, &meta, &meta).expect("eq");
    let lt_ab = lt_tensor_contiguous_f64(&a, &b, &meta, &meta).expect("lt");
    let lt_ba = lt_tensor_contiguous_f64(&b, &a, &meta, &meta).expect("lt rev");
    let neg_b = neg_tensor_contiguous_f64(&b, &meta).expect("neg");
    let add_a_negb =
        add_tensor_contiguous_f64(&a, &neg_b, &meta, &meta).expect("add(a, neg(b))");

    assert_eq!(add_ab.len(), numel, "add length");
    assert_eq!(sub_ab.len(), numel, "sub length");
    assert_eq!(mul_ab.len(), numel, "mul length");
    assert_eq!(div_ab.len(), numel, "div length");
    assert_eq!(eq_ab.len(), numel, "eq length");
    assert_eq!(lt_ab.len(), numel, "lt length");

    // Commutativity of add and mul (bit-exact for finite, NaN-NaN).
    let add_ba = add_tensor_contiguous_f64(&b, &a, &meta, &meta).expect("add rev");
    let mul_ba = mul_tensor_contiguous_f64(&b, &a, &meta, &meta).expect("mul rev");
    for i in 0..numel {
        if !add_ab[i].is_nan() || !add_ba[i].is_nan() {
            assert_eq!(
                add_ab[i].to_bits(),
                add_ba[i].to_bits(),
                "add commutativity broken at {i}: a+b={} b+a={}", add_ab[i], add_ba[i]
            );
        }
        if !mul_ab[i].is_nan() || !mul_ba[i].is_nan() {
            assert_eq!(
                mul_ab[i].to_bits(),
                mul_ba[i].to_bits(),
                "mul commutativity broken at {i}: a*b={} b*a={}", mul_ab[i], mul_ba[i]
            );
        }
    }

    // sub(a, b) == add(a, neg(b)) within ULP slack for finite cells.
    // Skip cells where either path produces NaN/inf.
    for i in 0..numel {
        if !sub_ab[i].is_finite() || !add_a_negb[i].is_finite() {
            continue;
        }
        let scale = sub_ab[i].abs().max(add_a_negb[i].abs()).max(1.0);
        assert!(
            (sub_ab[i] - add_a_negb[i]).abs() <= 4.0 * f64::EPSILON * scale,
            "sub vs add-neg drift at {i}: sub={}, add_neg={}", sub_ab[i], add_a_negb[i]
        );
    }

    // eq output is exactly 0.0 or 1.0.
    for (i, &v) in eq_ab.iter().enumerate() {
        assert!(
            v == 0.0 || v == 1.0,
            "eq[{i}] = {v} not in {{0.0, 1.0}}"
        );
    }
    // lt output is exactly 0.0 or 1.0.
    for (i, &v) in lt_ab.iter().enumerate() {
        assert!(
            v == 0.0 || v == 1.0,
            "lt[{i}] = {v} not in {{0.0, 1.0}}"
        );
    }

    // Trichotomy: for non-NaN cells, exactly one of lt(a,b), lt(b,a),
    // eq(a,b) is 1.0. Their sum is 1.0.
    for i in 0..numel {
        if a[i].is_nan() || b[i].is_nan() {
            continue;
        }
        let sum = lt_ab[i] + lt_ba[i] + eq_ab[i];
        assert!(
            (sum - 1.0).abs() < f64::EPSILON,
            "trichotomy broken at {i}: lt(a,b)={}, lt(b,a)={}, eq={} (sum={})",
            lt_ab[i], lt_ba[i], eq_ab[i], sum
        );
    }
});
