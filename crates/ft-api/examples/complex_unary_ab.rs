//! A/B for kgs4.92: complex transcendental unary ops (log/exp/sin/tanh/…) were
//! serial in complex_unary_fast + the complex-exp path. Now parallel above
//! PARALLEL_ELEMENTWISE_MIN. 1-thread vs full-pool == exact before/after (old serial
//! == 1-thread). Pure per-element maps → bit-identical (complex unit tests pass).
//!   cargo run -q --release -p ft-api --example complex_unary_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

macro_rules! timeit {
    ($s:expr, $z:expr, $reps:expr, $m:ident) => {{
        let _ = $s.$m($z).unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..$reps {
            let t = Instant::now();
            std::hint::black_box($s.$m($z).unwrap());
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    }};
}

fn run_once(numel: usize, reps: usize) -> (f64, f64, f64, f64) {
    let re: Vec<f64> = (0..numel)
        .map(|i| ((i * 2654435761usize) % 4001) as f64 * 0.001 - 2.0)
        .collect();
    let im: Vec<f64> = (0..numel)
        .map(|i| ((i * 40503usize) % 4001) as f64 * 0.001 - 2.0)
        .collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let rn = s.tensor_variable(re, vec![numel], false).unwrap();
    let imn = s.tensor_variable(im, vec![numel], false).unwrap();
    let z = s.tensor_complex(rn, imn).unwrap();
    let log = timeit!(s, z, reps, tensor_log);
    let exp = timeit!(s, z, reps, tensor_exp);
    let sin = timeit!(s, z, reps, tensor_sin);
    let tanh = timeit!(s, z, reps, tensor_tanh);
    (log, exp, sin, tanh)
}

fn main() {
    let nthreads = rayon::current_num_threads();
    let numel = 262_144usize;
    let p1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let (l1, e1, s1, t1) = p1.install(|| run_once(numel, 30));
    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    let (ln, en, sn, tn) = pn.install(|| run_once(numel, 30));
    println!("complex unary numel={numel}  OLD(1t) vs NEW({nthreads}t):");
    println!("  log : {l1:.3} -> {ln:.3} ms ({:.2}x)", l1 / ln);
    println!("  exp : {e1:.3} -> {en:.3} ms ({:.2}x)", e1 / en);
    println!("  sin : {s1:.3} -> {sn:.3} ms ({:.2}x)", s1 / sn);
    println!("  tanh: {t1:.3} -> {tn:.3} ms ({:.2}x)", t1 / tn);
}
