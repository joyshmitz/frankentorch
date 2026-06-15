//! Bench for kgs4.86: single-plane (batch=1) fft2 multicore scaling.
//! The second-to-last-dim (strided column) FFT and the final complex assembly of
//! a lone 2-D matrix used to run serially (column-parallel only engaged for
//! batch>=2); this times the production `tensor_fft2` on one large matrix under a
//! 1-thread vs full-pool rayon scope. Bit-exactness of the parallel path is proven
//! by the `fft2_parallel_match_serial_bit_exact` unit test.
//!   cargo run -q --release -p ft-api --example fft2_colpar_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn run_once(rows: usize, cols: usize, reps: usize) -> (f64, f64) {
    let n = rows * cols;
    let data: Vec<f64> = (0..n)
        .map(|i| ((i * 2654435761usize) % 1009) as f64 * 0.001 - 0.5)
        .collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let v = s
        .tensor_variable(data.clone(), vec![rows, cols], false)
        .unwrap();
    let _ = s.tensor_fft2(v).unwrap();
    let mut checksum = 0.0;
    let t = Instant::now();
    for _ in 0..reps {
        let v = s
            .tensor_variable(data.clone(), vec![rows, cols], false)
            .unwrap();
        let y = s.tensor_fft2(v).unwrap();
        checksum += y.0 as f64;
    }
    (t.elapsed().as_secs_f64() * 1e3 / reps as f64, checksum)
}

fn main() {
    for &(rows, cols, reps) in &[
        (1024usize, 1024usize, 6usize),
        (2048, 512, 6),
        (4096, 256, 6),
    ] {
        let pool1 = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let (t1, _) = pool1.install(|| run_once(rows, cols, reps));
        let pooln = rayon::ThreadPoolBuilder::new().build().unwrap();
        let nthreads = pooln.current_num_threads();
        let (tn, _) = pooln.install(|| run_once(rows, cols, reps));
        println!(
            "fft2 [{rows}x{cols}] batch=1: 1-thread {t1:.2}ms  full-pool({nthreads}t) {tn:.2}ms  =>  {:.2}x",
            t1 / tn,
        );
    }
}
