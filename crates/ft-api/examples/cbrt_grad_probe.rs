// Verifies tensor_cbrt forward value (libm-exact) plus 1st/2nd-order grad.
// cbrt(x): f=cbrt(x), f'=1/(3 cbrt(x)^2), f''=-2/(9 cbrt(x)^5).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn main() {
    let xs = vec![8.0f64, -27.0, 0.125, 64.0];
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let a = s.tensor_variable(xs.clone(), vec![xs.len()], true).unwrap();
    let y = s.tensor_cbrt(a).unwrap();
    let yv = s.tensor_values(y).unwrap();
    // first-order grad of sum(cbrt)
    let loss = s.tensor_sum(y).unwrap();
    let report = s.tensor_backward(loss).unwrap();
    let g = report.gradient(a).unwrap();
    println!("x     fwd            libm_cbrt      fwd_ok grad           grad_expected   grad_ok");
    let mut all_ok = true;
    for i in 0..xs.len() {
        let x = xs[i];
        let cb = libm::cbrt(x);
        let fwd_ok = yv[i].to_bits() == cb.to_bits();
        let gexp = 1.0 / (3.0 * cb * cb);
        let grad_ok = (g[i] - gexp).abs() < 1e-12 * gexp.abs().max(1.0);
        all_ok &= fwd_ok && grad_ok;
        println!(
            "{:>6} {:>14.9} {:>14.9} {:>6} {:>14.9} {:>14.9} {:>6}",
            x, yv[i], cb, fwd_ok, g[i], gexp, grad_ok
        );
    }
    println!("ALL_OK={all_ok}");
    assert!(all_ok, "cbrt forward/grad mismatch");
}
