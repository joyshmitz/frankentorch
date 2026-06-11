//! A/B: public f32 eigvalsh — native f32 (b6pem, f32 Householder reduction + f32
//! QL, ~half the memory traffic) vs the old f32->f64->f32 upcast path. The
//! symmetric tridiagonalization is memory-bandwidth-bound, so f32 should win.
//!   cargo run -q --release -p ft-api --example eigvalsh_f32_probe
use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
use std::time::Instant;

fn sym(n: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = (((i * 13 + j * 7) % 17) as f32) * 0.1 - 0.8;
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
        a[i * n + i] += n as f32;
    }
    a
}

fn main() {
    let it = 8;
    for &n in &[256usize, 512, 1024] {
        let a = sym(n);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let warm = s.tensor_variable_f32(a.clone(), vec![n, n], false).unwrap();
        let _ = s.tensor_linalg_eigvalsh(warm).unwrap();
        let t0 = Instant::now();
        for _ in 0..it {
            let t = s.tensor_variable_f32(a.clone(), vec![n, n], false).unwrap();
            std::hint::black_box(s.tensor_linalg_eigvalsh(t).unwrap());
        }
        let native = t0.elapsed().as_secs_f64() * 1e3 / it as f64;
        let t1 = Instant::now();
        for _ in 0..it {
            let t = s.tensor_variable_f32(a.clone(), vec![n, n], false).unwrap();
            let t64 = s.tensor_to_dtype(t, DType::F64).unwrap();
            let o64 = s.tensor_linalg_eigvalsh(t64).unwrap();
            std::hint::black_box(s.tensor_to_dtype(o64, DType::F32).unwrap());
        }
        let up = t1.elapsed().as_secs_f64() * 1e3 / it as f64;
        println!(
            "eigvalsh f32 {n}x{n}: native={native:.2}ms  f64-upcast={up:.2}ms  speedup={:.2}x",
            up / native
        );
    }
}
