use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::hint::black_box;
use std::time::Instant;
fn t<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..2 {
        f();
    }
    let mut b = f64::MAX;
    for _ in 0..6 {
        let s = Instant::now();
        f();
        b = b.min(s.elapsed().as_secs_f64() * 1000.0);
    }
    b
}
fn main() {
    for n in [512usize, 1024, 2048] {
        let mut a = vec![0.0f64; n * n];
        for r in 0..n {
            for c in 0..n {
                a[r * n + c] = (((r * 31 + c * 17) % 1000) as f64) * 0.001 - 0.5;
            }
            a[r * n + r] += n as f64;
        }
        let mut spd = vec![0.0f64; n * n];
        for r in 0..n {
            for c in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += a[r * n + k] * a[c * n + k];
                }
                spd[r * n + c] = s;
            }
            spd[r * n + r] += n as f64;
        }
        let chol = t(|| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(spd.clone(), vec![n, n], false).unwrap();
            black_box(s.tensor_cholesky(x, false).unwrap());
        });
        println!(
            "cholesky n={n}: {chol:.2} ms ({:.1} GFLOP/s)",
            (n as f64).powi(3) / 3.0 / (chol / 1000.0) / 1e9
        );
    }
}
