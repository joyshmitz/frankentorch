use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::hint::black_box;
use std::time::Instant;
fn t<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..2 {
        f();
    }
    let mut b = f64::MAX;
    for _ in 0..5 {
        let s = Instant::now();
        for _ in 0..3 {
            f();
        }
        b = b.min(s.elapsed().as_secs_f64() / 3.0 * 1000.0);
    }
    b
}
// Old quantile(linear, q=0.5): full sort of indices, then pick lo/hi, lerp.
fn old_quantile_mid(vals: &[f64]) -> f64 {
    let n = vals.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
    let fidx = 0.5 * (n - 1) as f64;
    let lo = fidx.floor() as usize;
    let hi = fidx.ceil() as usize;
    let frac = fidx - lo as f64;
    let l = vals[idx[lo]];
    let h = vals[idx[hi]];
    l + frac * (h - l)
}
fn main() {
    let n = 1_000_000usize;
    let data: Vec<f64> = (0..n)
        .map(|i| ((i * 2654435761usize) % 1_000_003) as f64)
        .collect();
    let old = t(|| {
        black_box(old_quantile_mid(black_box(&data)));
    });
    let new = t(|| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let tt = s.tensor_variable(data.clone(), vec![n], false).unwrap();
        black_box(s.tensor_quantile(tt, 0.5).unwrap());
    });
    println!(
        "quantile(0.5) n={n}: old(full-sort) {old:.3}ms | new(2x quickselect, incl session) {new:.3}ms | {:.2}x",
        old / new
    );
}
