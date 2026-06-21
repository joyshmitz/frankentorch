use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::outer_tensor_contiguous_f64;
use std::time::Instant;
fn main() {
    let nthreads = rayon::current_num_threads();
    let (m, n) = (4096usize, 4096);
    let a: Vec<f64> = (0..m).map(|i| (i % 100) as f64 * 0.01).collect();
    let b: Vec<f64> = (0..n).map(|i| (i % 97) as f64 * 0.01).collect();
    let am = TensorMeta::from_shape(vec![m], DType::F64, Device::Cpu);
    let bm = TensorMeta::from_shape(vec![n], DType::F64, Device::Cpu);
    let run = || {
        std::hint::black_box(outer_tensor_contiguous_f64(&a, &b, &am, &bm).unwrap());
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
        "outer [{m}]x[{n}] -> {}M: OLD(1t) {old:.2}ms NEW({nthreads}t) {new:.2}ms => {:.2}x",
        m * n / 1_000_000,
        old / new
    );
}
