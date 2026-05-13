#![no_main]

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 256;

fuzz_target!(|data: &[u8]| {
    if data.len() < 1 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let n = usize::from(1 + (data[0] % 16)); // 1..=16
    let body = &data[1..];

    if body.len() < 2 * n {
        return;
    }
    let re: Vec<f64> = (0..n)
        .map(|i| {
            let raw = body[i % body.len().max(1)] as i32;
            (raw - 128) as f64 / 20.0
        })
        .collect();
    let im: Vec<f64> = (0..n)
        .map(|i| {
            let raw = body[(n + i) % body.len().max(1)] as i32;
            (raw - 128) as f64 / 20.0
        })
        .collect();

    // Round-trip 1: complex(real(z), imag(z)) == z bit-exactly.
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let re_node = match s.tensor_variable(re.clone(), vec![n], false) {
        Ok(t) => t,
        Err(_) => return,
    };
    let im_node = match s.tensor_variable(im.clone(), vec![n], false) {
        Ok(t) => t,
        Err(_) => return,
    };
    let z = match s.tensor_complex(re_node, im_node) {
        Ok(t) => t,
        Err(_) => return,
    };
    // Now extract real and imag and re-pack.
    let re_back = match s.tensor_real(z) {
        Ok(t) => t,
        Err(_) => return,
    };
    let im_back = match s.tensor_imag(z) {
        Ok(t) => t,
        Err(_) => return,
    };
    let re_back_vals = s.tensor_values(re_back).expect("re_back vals");
    let im_back_vals = s.tensor_values(im_back).expect("im_back vals");
    assert_eq!(re_back_vals.len(), n);
    assert_eq!(im_back_vals.len(), n);
    for (i, (got, expected)) in re_back_vals.iter().zip(re.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "tensor_real(complex(re, im))[{i}] = {got}, expected re[{i}] = {expected}"
        );
    }
    for (i, (got, expected)) in im_back_vals.iter().zip(im.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "tensor_imag(complex(re, im))[{i}] = {got}, expected im[{i}] = {expected}"
        );
    }

    // Round-trip 2: view_as_complex(view_as_real(z)) == z bit-exactly.
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let re_node = s.tensor_variable(re.clone(), vec![n], false).expect("re_node");
    let im_node = s.tensor_variable(im.clone(), vec![n], false).expect("im_node");
    let z = match s.tensor_complex(re_node, im_node) {
        Ok(t) => t,
        Err(_) => return,
    };
    let real_view = match s.tensor_view_as_real(z) {
        Ok(t) => t,
        Err(_) => return,
    };
    let z_back = match s.tensor_view_as_complex(real_view) {
        Ok(t) => t,
        Err(_) => return,
    };
    // Compare via real/imag of both.
    let re_z_back = s.tensor_real(z_back).expect("re_z_back");
    let im_z_back = s.tensor_imag(z_back).expect("im_z_back");
    let re_z_back_vals = s.tensor_values(re_z_back).expect("re_z_back vals");
    let im_z_back_vals = s.tensor_values(im_z_back).expect("im_z_back vals");
    for (i, (got, expected)) in re_z_back_vals.iter().zip(re.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "view_as_complex(view_as_real(z)).real[{i}] = {got}, expected {expected}"
        );
    }
    for (i, (got, expected)) in im_z_back_vals.iter().zip(im.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "view_as_complex(view_as_real(z)).imag[{i}] = {got}, expected {expected}"
        );
    }
});
