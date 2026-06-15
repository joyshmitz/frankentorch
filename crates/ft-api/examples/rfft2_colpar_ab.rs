//! Bench: single-plane rfft2 multicore scaling (1-thread vs full pool).
//! (Self-contained; the kgs4.88 toggle scaffold was reverted.)
//!   cargo run -q --release -p ft-api --example rfft2_colpar_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn run_once(rows: usize, cols: usize, reps: usize) -> f64 {
    let data: Vec<f64> = (0..rows * cols)
        .map(|i| ((i * 2654435761usize) % 1009) as f64 * 0.001 - 0.5)
        .collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let v = s
        .tensor_variable(data.clone(), vec![rows, cols], false)
        .unwrap();
    let _ = s.tensor_rfft2(v).unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let v = s
            .tensor_variable(data.clone(), vec![rows, cols], false)
            .unwrap();
        let t = Instant::now();
        std::hint::black_box(s.tensor_rfft2(v).unwrap());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() {
    for &(rows, cols) in &[(1024usize, 1024usize), (2048, 512)] {
        let p1 = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let t1 = p1.install(|| run_once(rows, cols, 15));
        let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
        let nthreads = pn.current_num_threads();
        let tn = pn.install(|| run_once(rows, cols, 15));
        println!(
            "rfft2 [{rows}x{cols}]: 1t {t1:.2}ms  {nthreads}t {tn:.2}ms  => {:.2}x",
            t1 / tn
        );
    }
}
