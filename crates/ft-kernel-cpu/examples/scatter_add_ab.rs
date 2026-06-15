//! A/B for kgs4.96b: scatter_add (embedding/segment-sum backward) was a serial
//! sweep. Distinct (outer,inner) columns touch disjoint outputs, so parallelize over
//! columns (each scatters its `r` in order = bit-exact, disjoint writes = race-free).
//! 1-thread vs full-pool == exact before/after. Bit-exactness proven by
//! scatter_add_parallel_matches_serial_with_collisions_bit_exact.
//!   cargo run -q --release -p ft-kernel-cpu --example scatter_add_ab
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::scatter_add_tensor_contiguous_f64;
use std::time::Instant;

fn main() {
    let nthreads = rayon::current_num_threads();
    // embedding-backward style: out [num_emb, m] <- scatter rows of src [n, m] (dim 0).
    let (num_emb, m, n) = (512usize, 512usize, 4096usize);
    let input: Vec<f64> = vec![0.0; num_emb * m];
    let src: Vec<f64> = (0..n * m).map(|i| (i % 1000) as f64 * 0.001).collect();
    let index: Vec<f64> = (0..n * m)
        .map(|i| (i.wrapping_mul(2654435761) % num_emb) as f64)
        .collect();
    let meta = TensorMeta::from_shape(vec![num_emb, m], DType::F64, Device::Cpu);
    let idx_meta = TensorMeta::from_shape(vec![n, m], DType::F64, Device::Cpu);
    let run = || {
        std::hint::black_box(
            scatter_add_tensor_contiguous_f64(&input, &meta, 0, &index, &idx_meta, &src).unwrap(),
        );
    };
    let bench = |reps: usize| {
        run();
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            run();
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    };
    let p1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let old = p1.install(|| bench(30));
    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    let new = pn.install(|| bench(30));
    println!(
        "scatter_add [4096,512]->[512,512] dim0: OLD(1t) {old:.2}ms  NEW({nthreads}t) {new:.2}ms  =>  {:.2}x",
        old / new
    );
}
