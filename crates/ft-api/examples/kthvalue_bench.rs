use std::hint::black_box;
use std::time::Instant;
fn t<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..2 {
        f();
    }
    let mut b = f64::MAX;
    for _ in 0..6 {
        let s = Instant::now();
        for _ in 0..3 {
            f();
        }
        b = b.min(s.elapsed().as_secs_f64() / 3.0 * 1000.0);
    }
    b
}
fn old_nq(data: &[f64], q: f64) -> f64 {
    // filter NaN, full sort, linear interp
    let mut nn: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
    nn.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = nn.len();
    let idx = q * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    nn[lo] * (1.0 - frac) + nn[hi] * frac
}
fn new_nq(data: &[f64], q: f64) -> f64 {
    // filter NaN, quickselect lo/hi
    let mut nn: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = nn.len();
    let idx = q * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    let kv = |nn: &mut [f64], k: usize| -> f64 {
        let (_, v, _) = nn.select_nth_unstable_by(k, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        *v
    };
    let l = kv(&mut nn, lo);
    let h = kv(&mut nn, hi);
    l * (1.0 - frac) + h * frac
}
fn main() {
    let n = 1_000_000usize;
    let data: Vec<f64> = (0..n)
        .map(|i| {
            if i % 50 == 0 {
                f64::NAN
            } else {
                ((i * 2654435761usize) % 1_000_003) as f64
            }
        })
        .collect();
    let old = t(|| {
        black_box(old_nq(black_box(&data), 0.5));
    });
    let new = t(|| {
        black_box(new_nq(black_box(&data), 0.5));
    });
    println!(
        "nanquantile n={n}: old(filter+sort) {old:.3}ms | new(filter+quickselect) {new:.3}ms | {:.2}x",
        old / new
    );
}
