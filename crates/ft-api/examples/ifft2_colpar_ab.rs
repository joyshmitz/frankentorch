//! Bench for kgs4.87: single-plane (batch=1) ifft2 multicore scaling.
//! The strided second-to-last-dim inverse transform, the complex deinterleave of
//! the input, and the final norm-scale + Complex128 assembly of a lone 2-D matrix
//! used to run serially; this times the production `tensor_ifft2` (input = fft2 of
//! a real plane) under a 1-thread vs full-pool rayon scope. Bit-exactness of the
//! parallel path is proven by `ifft2_rfft2_parallel_match_serial_bit_exact`.
//!   cargo run -q --release -p ft-api --example ifft2_colpar_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn run_once(rows: usize, cols: usize, reps: usize) -> f64 {
    let n = rows * cols;
    let data: Vec<f64> = (0..n)
        .map(|i| ((i * 2654435761usize) % 1009) as f64 * 0.001 - 0.5)
        .collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let v = s
        .tensor_variable(data.clone(), vec![rows, cols], false)
        .unwrap();
    let spec = s.tensor_fft2(v).unwrap();
    let _ = s.tensor_ifft2(spec).unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let y = s.tensor_ifft2(spec).unwrap();
        std::hint::black_box(y.0);
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() {
    for &(rows, cols, reps) in &[
        (1024usize, 1024usize, 20usize),
        (2048, 512, 20),
        (4096, 256, 20),
    ] {
        let pool1 = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let t1 = pool1.install(|| run_once(rows, cols, reps));
        let pooln = rayon::ThreadPoolBuilder::new().build().unwrap();
        let nthreads = pooln.current_num_threads();
        let tn = pooln.install(|| run_once(rows, cols, reps));
        println!(
            "ifft2 [{rows}x{cols}] batch=1: 1-thread {t1:.2}ms  full-pool({nthreads}t) {tn:.2}ms  =>  {:.2}x",
            t1 / tn,
        );
    }
}
