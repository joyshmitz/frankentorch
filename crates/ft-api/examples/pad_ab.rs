//! A/B for kgs4.96: tensor_pad_mode (reflect/replicate/circular) built its gather
//! index vector with a serial per-output loop (decompose flat index, map each dim
//! through reflect/clamp/wrap). That per-element index math is compute-bound, now
//! parallelized. 1-thread vs full-pool == exact before/after (old serial == 1-thread).
//! Bit-exactness proven by the pad_reflect_* / pad_circular_* unit tests.
//!   cargo run -q --release -p ft-api --example pad_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn bench(name: &str, mode: &str, reps: usize) {
    let (h, w) = (2048usize, 2048usize);
    let pad = [64usize, 64, 64, 64]; // pad last two dims by 64 each side
    let data: Vec<f64> = (0..h * w).map(|i| (i % 1000) as f64 * 0.001).collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let run = |s: &mut FrankenTorchSession| {
        let v = s.tensor_variable(data.clone(), vec![h, w], false).unwrap();
        std::hint::black_box(s.tensor_pad_mode(v, &pad, mode, 0.0).unwrap());
    };
    run(&mut s);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        run(&mut s);
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("  {name:<22}: {best:.3} ms");
}

fn main() {
    let nthreads = rayon::current_num_threads();
    for mode in ["reflect", "replicate"] {
        let p1 = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        p1.install(|| bench(&format!("{mode} OLD(1t)"), mode, 30));
        let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
        pn.install(|| bench(&format!("{mode} NEW({nthreads}t)"), mode, 30));
    }
}
