//! 1q8x re-measure: is the f32 scan PATH genuinely ~20x slower than f64?
//! Times the real production tape path (no grad) for cumsum + cumprod at
//! [8192,1024] dim=1, f32 vs f64, in ONE process.
//!   cargo run -q --release -p ft-api --example scan_f32_vs_f64
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    let nthreads = rayon::current_num_threads();
    let (rows, cols) = (8192usize, 1024);
    let n = rows * cols;
    let d64: Vec<f64> = (0..n).map(|i| 1.0 + ((i % 7) as f64) * 1e-3).collect();
    let d32: Vec<f32> = d64.iter().map(|&x| x as f32).collect();

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let v64 = s.tensor_variable(d64.clone(), vec![rows, cols], false).unwrap();
    let v32 = s.tensor_variable_f32(d32.clone(), vec![rows, cols], false).unwrap();

    macro_rules! bench {
        ($name:expr, $call:expr) => {{
            std::hint::black_box($call); // warm
            let mut best = f64::INFINITY;
            for _ in 0..15 {
                let t = Instant::now();
                std::hint::black_box($call);
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            println!("{:<18} {:.2}ms", $name, best);
            best
        }};
    }

    println!("scan [{rows},{cols}] dim=1  ({nthreads} threads)");
    let cs64 = bench!("cumsum f64", s.tensor_cumsum(v64, 1).unwrap());
    let cs32 = bench!("cumsum f32", s.tensor_cumsum(v32, 1).unwrap());
    let cp64 = bench!("cumprod f64", s.tensor_cumprod(v64, 1).unwrap());
    let cp32 = bench!("cumprod f32", s.tensor_cumprod(v32, 1).unwrap());
    println!("cumsum  f32/f64 = {:.2}x   cumprod f32/f64 = {:.2}x", cs32 / cs64, cp32 / cp64);
}
