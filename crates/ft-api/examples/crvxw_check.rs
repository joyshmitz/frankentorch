// Regression probe: ft cholesky_solve produces torch's DENSE factor gradient
// (off-triangle entries nonzero), row-major, lower & upper. frankentorch-crvxw.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn run(name: &str, l: Vec<f64>, n: usize, b: Vec<f64>, upper: bool, g1: &[f64]) -> bool {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let lid = s.tensor_variable(l, vec![n, n], true).unwrap();
    let bid = s.tensor_variable(b, vec![n, 1], false).unwrap();
    let x = s.tensor_cholesky_solve(bid, lid, upper).unwrap();
    let loss = s.tensor_sum(x).unwrap();
    let rep = s.tensor_backward(loss).unwrap();
    let g = rep.gradient(lid).unwrap();
    let ok = (0..n * n).all(|i| (g[i] - g1[i]).abs() < 1e-9);
    println!("{name} MATCH={ok} ft={:?}", g);
    ok
}

fn main() {
    let mut all = true;
    // lower 2x2 (torch row-major flatten)
    all &= run(
        "LO2",
        vec![std::f64::consts::SQRT_2, 0.0, 0.35355339, 1.17260394],
        2,
        vec![1.0, 2.0],
        false,
        &[-0.385694611, -0.658984036, -1.285648695, -1.628078197],
    );
    // upper 2x2
    all &= run(
        "UP2",
        vec![std::f64::consts::SQRT_2, 0.35355339, 0.0, 1.17260394],
        2,
        vec![1.0, 2.0],
        true,
        &[-0.385694611, -1.285648695, -0.658984036, -1.628078197],
    );
    // lower 3x3
    all &= run(
        "LO3",
        vec![2.0, 0.0, 0.0, 0.5, 1.5, 0.0, 0.3, 0.4, 1.2],
        3,
        vec![1.0, 2.0, 3.0],
        false,
        &[
            -0.023341049,
            -0.109375,
            -0.21169303,
            -0.274691358,
            -0.416666667,
            -0.605195473,
            -1.053240741,
            -1.354166667,
            -1.793016975,
        ],
    );
    println!("ALL_MATCH={all}");
    assert!(all, "cholesky_solve dense grad mismatch vs torch");
}
