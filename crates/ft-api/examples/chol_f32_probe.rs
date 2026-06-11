//! A/B: public f32 cholesky / cholesky_solve / lu_solve — native kernel (b3o90)
//! vs the f64-upcast reference (cast f32->f64, solve, cast back).
//!   cargo run -q --release -p ft-api --example chol_f32_probe
use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
use std::time::Instant;

fn spd(n: usize) -> Vec<f32> {
    // A = M M^T + n I  (symmetric positive definite, well-conditioned).
    let m: Vec<f32> = (0..n * n)
        .map(|i| ((i * 2654435761usize) % 97) as f32 * 0.01 - 0.48)
        .collect();
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += m[i * n + k] * m[j * n + k];
            }
            a[i * n + j] = acc;
        }
        a[i * n + i] += n as f32;
    }
    a
}

fn main() {
    let it = 10;
    for &n in &[256usize, 512] {
        let a = spd(n);
        let b: Vec<f32> = (0..n).map(|i| (i % 7) as f32 + 1.0).collect();
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);

        // ---- cholesky ----
        let warm = s.tensor_variable_f32(a.clone(), vec![n, n], false).unwrap();
        let _ = s.tensor_linalg_cholesky(warm, false).unwrap();
        let t0 = Instant::now();
        for _ in 0..it {
            let t = s.tensor_variable_f32(a.clone(), vec![n, n], false).unwrap();
            std::hint::black_box(s.tensor_linalg_cholesky(t, false).unwrap());
        }
        let native = t0.elapsed().as_secs_f64() * 1e3 / it as f64;
        let t1 = Instant::now();
        for _ in 0..it {
            let t = s.tensor_variable_f32(a.clone(), vec![n, n], false).unwrap();
            let t64 = s.tensor_to_dtype(t, DType::F64).unwrap();
            let o64 = s.tensor_linalg_cholesky(t64, false).unwrap();
            std::hint::black_box(s.tensor_to_dtype(o64, DType::F32).unwrap());
        }
        let up = t1.elapsed().as_secs_f64() * 1e3 / it as f64;
        println!(
            "cholesky      f32 {n}x{n}: native={native:.2}ms  f64-upcast={up:.2}ms  speedup={:.2}x",
            up / native
        );

        // factor (f32) for the solves
        let af = s.tensor_variable_f32(a.clone(), vec![n, n], false).unwrap();
        let lf = s.tensor_linalg_cholesky(af, false).unwrap();

        // ---- cholesky_solve ----
        let bw = s.tensor_variable_f32(b.clone(), vec![n, 1], false).unwrap();
        let _ = s.tensor_cholesky_solve(bw, lf, false).unwrap();
        let t0 = Instant::now();
        for _ in 0..it {
            let bb = s.tensor_variable_f32(b.clone(), vec![n, 1], false).unwrap();
            std::hint::black_box(s.tensor_cholesky_solve(bb, lf, false).unwrap());
        }
        let native = t0.elapsed().as_secs_f64() * 1e3 / it as f64;
        // reference = the TRUE old path: the f32 triangular_solve composition
        // (L^-1 then L^-T), exactly what tensor_cholesky_solve ran before b3o90.
        let t1 = Instant::now();
        for _ in 0..it {
            let bb = s.tensor_variable_f32(b.clone(), vec![n, 1], false).unwrap();
            let y = s.tensor_triangular_solve(lf, bb, false).unwrap();
            let lt = s.tensor_transpose(lf, 0, 1).unwrap();
            std::hint::black_box(s.tensor_triangular_solve(lt, y, true).unwrap());
        }
        let up = t1.elapsed().as_secs_f64() * 1e3 / it as f64;
        println!(
            "cholesky_solve f32 {n}x{n}: native={native:.2}ms  old-compose={up:.2}ms  speedup={:.2}x",
            up / native
        );

        // ---- lu_solve ----
        let alu = s.tensor_variable_f32(a.clone(), vec![n, n], false).unwrap();
        let (lu, piv) = s.tensor_lu_factor(alu).unwrap();
        let bw = s.tensor_variable_f32(b.clone(), vec![n, 1], false).unwrap();
        let _ = s.tensor_lu_solve(lu, &piv, bw).unwrap();
        let t0 = Instant::now();
        for _ in 0..it {
            let bb = s.tensor_variable_f32(b.clone(), vec![n, 1], false).unwrap();
            std::hint::black_box(s.tensor_lu_solve(lu, &piv, bb).unwrap());
        }
        let native = t0.elapsed().as_secs_f64() * 1e3 / it as f64;
        let lu64 = s.tensor_to_dtype(lu, DType::F64).unwrap();
        let t1 = Instant::now();
        for _ in 0..it {
            let bb = s.tensor_variable_f32(b.clone(), vec![n, 1], false).unwrap();
            let b64 = s.tensor_to_dtype(bb, DType::F64).unwrap();
            let x64 = s.tensor_lu_solve(lu64, &piv, b64).unwrap();
            std::hint::black_box(s.tensor_to_dtype(x64, DType::F32).unwrap());
        }
        let up = t1.elapsed().as_secs_f64() * 1e3 / it as f64;
        println!(
            "lu_solve      f32 {n}x{n}: native={native:.2}ms  f64-upcast={up:.2}ms  speedup={:.2}x",
            up / native
        );
    }
}
