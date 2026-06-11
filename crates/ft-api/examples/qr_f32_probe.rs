//! A/B: public f32 linalg_qr — native f32 QR (b3o90, qr_contiguous_f32, blocked
//! compact-WY sgemm) vs the old f32->f64->f32 upcast path (f64 blocked QR).
//!   cargo run -q --release -p ft-api --example qr_f32_probe
use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
use std::time::Instant;

fn main() {
    let it = 10;
    for &n in &[256usize, 512] {
        let a: Vec<f32> = (0..n * n)
            .map(|i| (((i * 2654435761usize) % 101) as f32) * 0.02 - 1.0)
            .collect();
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);

        let warm = s.tensor_variable_f32(a.clone(), vec![n, n], false).unwrap();
        let _ = s.tensor_linalg_qr(warm, true).unwrap();
        let t0 = Instant::now();
        for _ in 0..it {
            let t = s.tensor_variable_f32(a.clone(), vec![n, n], false).unwrap();
            std::hint::black_box(s.tensor_linalg_qr(t, true).unwrap());
        }
        let native = t0.elapsed().as_secs_f64() * 1e3 / it as f64;
        // reference = old path: cast f32->f64, QR, cast Q/R back.
        let t1 = Instant::now();
        for _ in 0..it {
            let t = s.tensor_variable_f32(a.clone(), vec![n, n], false).unwrap();
            let t64 = s.tensor_to_dtype(t, DType::F64).unwrap();
            let (q, r) = s.tensor_linalg_qr(t64, true).unwrap();
            std::hint::black_box(s.tensor_to_dtype(q, DType::F32).unwrap());
            std::hint::black_box(s.tensor_to_dtype(r, DType::F32).unwrap());
        }
        let up = t1.elapsed().as_secs_f64() * 1e3 / it as f64;
        println!(
            "linalg_qr f32 {n}x{n}: native={native:.2}ms  f64-upcast={up:.2}ms  speedup={:.2}x",
            up / native
        );
    }
}
