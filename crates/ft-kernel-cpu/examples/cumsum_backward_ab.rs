//! Anchored A/B for cumsum_backward lane parallelization. Calls the PRODUCTION
//! kernel under a 1-thread pool (serial) vs the full pool (parallel) in ONE process,
//! asserts bit-for-bit equality, and reports the ratio. cumprod (forward) lane-parallel
//! is the proven sibling; cumsum_backward is the per-lane reverse cumsum.
//!   cargo run -q --release -p ft-kernel-cpu --example cumsum_backward_ab
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::cumsum_backward_tensor_contiguous_f64;

fn run_best(threads: usize, g: &[f64], meta: &TensorMeta) -> (f64, Vec<f64>) {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| {
        let out = cumsum_backward_tensor_contiguous_f64(g, meta, 1).unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..20 {
            let t = std::time::Instant::now();
            std::hint::black_box(cumsum_backward_tensor_contiguous_f64(g, meta, 1).unwrap());
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        (best, out)
    })
}

fn main() {
    let (rows, cols) = (8192usize, 1024);
    let n = rows * cols;
    let g: Vec<f64> = (0..n).map(|i| 1.0 + ((i % 13) as f64) * 1e-3).collect();
    let meta = TensorMeta::from_shape(vec![rows, cols], DType::F64, Device::Cpu);
    let nthreads = rayon::current_num_threads();

    let (ser, out_ser) = run_best(1, &g, &meta);
    let (par, out_par) = run_best(nthreads.max(2), &g, &meta);
    assert_eq!(out_ser.len(), out_par.len());
    for (a, b) in out_ser.iter().zip(out_par.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "parallel != serial bit-exact");
    }
    println!(
        "cumsum_backward [{rows},{cols}] dim=1 (bit-exact OK): serial(1t) {ser:.2}ms  parallel({nthreads}t) {par:.2}ms  =>  {:.2}x",
        ser / par
    );
}
