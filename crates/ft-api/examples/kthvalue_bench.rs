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
        for _ in 0..5 {
            f();
        }
        b = b.min(s.elapsed().as_secs_f64() / 5.0 * 1000.0);
    }
    b
}
// Old approach: full sort of indices by value.
fn old_kth(vals: &[f64], k: usize) -> usize {
    let mut indices: Vec<usize> = (0..vals.len()).collect();
    indices.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
    indices[k - 1]
}
fn main() {
    let n = 1_000_000usize;
    let data: Vec<f64> = (0..n)
        .map(|i| ((i * 2654435761usize) % 1_000_003) as f64)
        .collect();
    let m = n / 2;
    let old = t(|| {
        black_box(old_kth(black_box(&data), m));
    });
    let new = t(|| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let tt = s.tensor_variable(data.clone(), vec![n], false).unwrap();
        black_box(s.tensor_kthvalue(tt, m).unwrap());
    });
    println!(
        "kthvalue n={n}: old(full-sort) {old:.3}ms | new(quickselect, incl session) {new:.3}ms | {:.2}x",
        old / new
    );
}
