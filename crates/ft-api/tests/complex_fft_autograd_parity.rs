//! Regression: complex / FFT / IFFT autograd backward parity vs PyTorch 2.12.
//!
//! frankentorch-ng1hw closed the "last differentiable gaps" — the real<->complex
//! bridge ops and the FFT family backward. That surface was only exercised by the
//! `complex_autograd_probe` example (manual eyeball vs torch), with NO CI test and
//! a stale golden comment on the ifft(n=4) chain. This locks the verified torch
//! goldens so a regression (or a future "fix" toward a wrong golden) fails loudly.
//!
//! Goldens generated with torch 2.12.0+cpu (f64), e.g. for chain G:
//!   xr=[1,2,3,4]; xi=[.5,1,1.5,2]; y=ifft(complex(xr,xi), n=4)
//!   loss=sum(real(y)*[.1,.2,.3,.4] + imag(y)*[.5,1,1.5,2]); loss.backward()
//!   -> xr.grad=[0.25,-0.3,-0.05,0.2]; xi.grad=[1.25,-0.2,-0.25,-0.3]

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn assert_close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-9,
            "{label}[{i}]: got {g}, want {w} (torch golden)"
        );
    }
}

fn s() -> FrankenTorchSession {
    FrankenTorchSession::new(ExecutionMode::Strict)
}

/// Chain A: real/imag bridge — loss = sum(real(z)*2 + imag(z)*3).
#[test]
fn complex_real_imag_bridge_backward_matches_torch() {
    let mut s = s();
    let re = s
        .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], true)
        .unwrap();
    let im = s
        .tensor_variable(vec![0.5, 1.5, 2.5, 3.5], vec![4], true)
        .unwrap();
    let z = s.tensor_complex(re, im).unwrap();
    let rp = s.tensor_real(z).unwrap();
    let ip = s.tensor_imag(z).unwrap();
    let a = s.tensor_mul_scalar(rp, 2.0).unwrap();
    let b = s.tensor_mul_scalar(ip, 3.0).unwrap();
    let sum = s.tensor_add(a, b).unwrap();
    let out = s.tensor_sum(sum).unwrap();
    let rep = s.tensor_backward(out).unwrap();
    assert_close(
        "A re",
        s.tensor_gradient(&rep, re).unwrap(),
        &[2.0, 2.0, 2.0, 2.0],
    );
    assert_close(
        "A im",
        s.tensor_gradient(&rep, im).unwrap(),
        &[3.0, 3.0, 3.0, 3.0],
    );
}

/// Chain B: abs(z) — grad re = re/|z|, im = im/|z|.
#[test]
fn complex_abs_backward_matches_torch() {
    let mut s = s();
    let re = s
        .tensor_variable(vec![3.0, 1.0, -2.0, 0.5], vec![4], true)
        .unwrap();
    let im = s
        .tensor_variable(vec![4.0, 1.0, 2.0, -0.5], vec![4], true)
        .unwrap();
    let z = s.tensor_complex(re, im).unwrap();
    let mag = s.tensor_abs(z).unwrap();
    let out = s.tensor_sum(mag).unwrap();
    let rep = s.tensor_backward(out).unwrap();
    assert_close(
        "B re",
        s.tensor_gradient(&rep, re).unwrap(),
        &[0.6, 0.707106781, -0.707106781, 0.707106781],
    );
    assert_close(
        "B im",
        s.tensor_gradient(&rep, im).unwrap(),
        &[0.8, 0.707106781, 0.707106781, -0.707106781],
    );
}

/// Chain D: real -> fft (no resize) -> weighted real/imag loss.
#[test]
fn fft_backward_matches_torch() {
    let mut s = s();
    let x = s
        .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], true)
        .unwrap();
    let y = s.tensor_fft(x, None).unwrap();
    let re = s.tensor_real(y).unwrap();
    let im = s.tensor_imag(y).unwrap();
    let a = s
        .tensor_variable(vec![0.1, 0.2, 0.3, 0.4], vec![4], false)
        .unwrap();
    let b = s
        .tensor_variable(vec![0.5, 0.6, 0.7, 0.8], vec![4], false)
        .unwrap();
    let ra = s.tensor_mul(re, a).unwrap();
    let ib = s.tensor_mul(im, b).unwrap();
    let sum = s.tensor_add(ra, ib).unwrap();
    let out = s.tensor_sum(sum).unwrap();
    let rep = s.tensor_backward(out).unwrap();
    assert_close(
        "D fft",
        s.tensor_gradient(&rep, x).unwrap(),
        &[1.0, 0.0, -0.2, -0.4],
    );
}

/// Chains E/F: fft with zero-padding (n=6) and truncation (n=2).
#[test]
fn fft_resize_backward_matches_torch() {
    for (label, n, golden) in [
        (
            "E pad n=6",
            6usize,
            vec![2.1, 2.298076211, 0.566025404, -0.3],
        ),
        ("F trunc n=2", 2usize, vec![0.3, -0.1, 0.0, 0.0]),
    ] {
        let mut s = s();
        let x = s
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], true)
            .unwrap();
        let y = s.tensor_fft(x, Some(n)).unwrap();
        let re = s.tensor_real(y).unwrap();
        let im = s.tensor_imag(y).unwrap();
        let av: Vec<f64> = (1..=n).map(|k| k as f64 * 0.1).collect();
        let bv: Vec<f64> = (1..=n).map(|k| k as f64 * 0.5).collect();
        let a = s.tensor_variable(av, vec![n], false).unwrap();
        let b = s.tensor_variable(bv, vec![n], false).unwrap();
        let ra = s.tensor_mul(re, a).unwrap();
        let ib = s.tensor_mul(im, b).unwrap();
        let sum = s.tensor_add(ra, ib).unwrap();
        let out = s.tensor_sum(sum).unwrap();
        let rep = s.tensor_backward(out).unwrap();
        assert_close(label, s.tensor_gradient(&rep, x).unwrap(), &golden);
    }
}

/// Chains G/H: complex -> ifft(n) -> weighted real/imag loss.
/// Chain G (n=4, exact) is the case whose probe golden comment was stale.
#[test]
fn ifft_backward_matches_torch() {
    for (label, n, gr, gi) in [
        (
            "G ifft n=4",
            4usize,
            vec![0.25, -0.3, -0.05, 0.2],
            vec![1.25, -0.2, -0.25, -0.3],
        ),
        (
            "H ifft pad n=6",
            6usize,
            vec![0.35, -0.483012702, -0.194337567, -0.05],
            vec![1.75, -0.16339746, -0.221132487, -0.25],
        ),
    ] {
        let mut s = s();
        let xr = s
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], true)
            .unwrap();
        let xi = s
            .tensor_variable(vec![0.5, 1.0, 1.5, 2.0], vec![4], true)
            .unwrap();
        let z = s.tensor_complex(xr, xi).unwrap();
        let y = s.tensor_ifft(z, Some(n)).unwrap();
        let re = s.tensor_real(y).unwrap();
        let im = s.tensor_imag(y).unwrap();
        let av: Vec<f64> = (1..=n).map(|k| k as f64 * 0.1).collect();
        let bv: Vec<f64> = (1..=n).map(|k| k as f64 * 0.5).collect();
        let a = s.tensor_variable(av, vec![n], false).unwrap();
        let b = s.tensor_variable(bv, vec![n], false).unwrap();
        let ra = s.tensor_mul(re, a).unwrap();
        let ib = s.tensor_mul(im, b).unwrap();
        let sum = s.tensor_add(ra, ib).unwrap();
        let out = s.tensor_sum(sum).unwrap();
        let rep = s.tensor_backward(out).unwrap();
        assert_close(
            &format!("{label} xr"),
            s.tensor_gradient(&rep, xr).unwrap(),
            &gr,
        );
        assert_close(
            &format!("{label} xi"),
            s.tensor_gradient(&rep, xi).unwrap(),
            &gi,
        );
    }
}

/// Chain C: view_as_complex -> view_as_real round-trip is differentiable.
#[test]
fn view_as_complex_real_roundtrip_backward_matches_torch() {
    let mut s = s();
    let rr = s
        .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], true)
        .unwrap();
    let z = s.tensor_view_as_complex(rr).unwrap();
    let r2 = s.tensor_view_as_real(z).unwrap();
    let w = s
        .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], false)
        .unwrap();
    let prod = s.tensor_mul(r2, w).unwrap();
    let out = s.tensor_sum(prod).unwrap();
    let rep = s.tensor_backward(out).unwrap();
    assert_close(
        "C",
        s.tensor_gradient(&rep, rr).unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
}
