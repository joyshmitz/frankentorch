//! A/B for kgs4.98: bessel i0e/i1/i1e forward maps were serial; now routed through
//! par_map_f64 (compute-bound continued-fraction/series). 1-thread vs full-pool ==
//! exact before/after (old serial == 1-thread). Pure per-element map → bit-identical.
//!   cargo run -q --release -p ft-api --example bessel_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

macro_rules! timeit {
    ($s:expr, $v:expr, $reps:expr, $m:ident) => {{
        let _ = $s.$m($v).unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..$reps {
            let t = Instant::now();
            std::hint::black_box($s.$m($v).unwrap());
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    }};
}

fn run_once(numel: usize, reps: usize) -> (f64, f64, f64) {
    let data: Vec<f64> = (0..numel).map(|i| (i % 1000) as f64 * 0.01).collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let v = s.tensor_variable(data, vec![numel], false).unwrap();
    (
        timeit!(s, v, reps, tensor_i0e),
        timeit!(s, v, reps, tensor_i1),
        timeit!(s, v, reps, tensor_i1e),
    )
}

fn main() {
    let nthreads = rayon::current_num_threads();
    let numel = 262_144usize;
    let p1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let (a1, b1, c1) = p1.install(|| run_once(numel, 30));
    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    let (an, bn, cn) = pn.install(|| run_once(numel, 30));
    println!("bessel numel={numel}  OLD(1t) vs NEW({nthreads}t):");
    println!("  i0e : {a1:.3} -> {an:.3} ms ({:.2}x)", a1 / an);
    println!("  i1  : {b1:.3} -> {bn:.3} ms ({:.2}x)", b1 / bn);
    println!("  i1e : {c1:.3} -> {cn:.3} ms ({:.2}x)", c1 / cn);
}
