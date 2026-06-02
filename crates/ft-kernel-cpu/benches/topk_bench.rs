//! topk-along-dim bench (compute-bound per-lane select, parallelized over lanes).
//!   baseline:  rch exec -- env RAYON_NUM_THREADS=1 cargo bench -p ft-kernel-cpu --bench topk_bench
//!   optimized: rch exec -- cargo bench -p ft-kernel-cpu --bench topk_bench

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::topk_tensor_contiguous_f64;

fn bench_topk(c: &mut Criterion) {
    let (rows, cols, k) = (8192usize, 1024usize, 50usize);
    let data: Vec<f64> = (0..rows * cols)
        .map(|i| ((i * 2654435761usize) % 100003) as f64 * 0.001)
        .collect();
    let meta = TensorMeta::from_shape(vec![rows, cols], DType::F64, Device::Cpu);
    c.bench_function("topk_f64_8192x1024_k50_dim1", |b| {
        b.iter(|| {
            black_box(
                topk_tensor_contiguous_f64(black_box(&data), &meta, k, 1, true, true)
                    .expect("valid topk input"),
            )
        })
    });
}

criterion_group!(benches, bench_topk);
criterion_main!(benches);
