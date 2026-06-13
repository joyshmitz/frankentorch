use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
fn main() {
    let xs = vec![1.0f64, 1.0000001, 1.1, 1.5, 2.0, 3.7, 100.0];
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let a = s.tensor_variable(xs.clone(), vec![xs.len()], false).unwrap();
    let oop_id = s.tensor_acosh(a).unwrap();
    let oop = s.tensor_values(oop_id).unwrap();
    let b = s.tensor_variable(xs.clone(), vec![xs.len()], false).unwrap();
    s.tensor_acosh_(b).unwrap();
    let ip = s.tensor_values(b).unwrap();
    let mut fwd_ok = true;
    let mut ipoop_ok = true;
    for i in 0..xs.len() {
        if oop[i].to_bits() != xs[i].acosh().to_bits() { fwd_ok = false; }
        if oop[i].to_bits() != ip[i].to_bits() { ipoop_ok = false; }
    }
    println!("FWD_LIBM_EXACT={fwd_ok} INPLACE_EQ_OOP={ipoop_ok}");
    let mut s2 = FrankenTorchSession::new(ExecutionMode::Strict);
    let x1 = s2.tensor_variable(vec![1.0, 1.5, 2.0], vec![3], true).unwrap();
    let y = s2.tensor_acosh(x1).unwrap();
    let loss = s2.tensor_sum(y).unwrap();
    let rep = s2.tensor_backward(loss).unwrap();
    let g = rep.gradient(x1).unwrap();
    println!("g1@[1,1.5,2]={:?} (torch inf,0.894427,0.577350)", g);
    assert!(fwd_ok && ipoop_ok, "acosh forward parity broken");
    assert!(g[0].is_infinite() && g[0] > 0.0, "acosh g1@1 should be +inf");
}
