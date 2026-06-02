//! Special-function (lgamma / digamma) elementwise throughput. These are
//! compute-bound per element (libm::lgamma; digamma recurrence + series) and are
//! evaluated across the rayon pool for large tensors. Toggle:
//!   baseline:  rch exec -- env RAYON_NUM_THREADS=1 cargo bench -p ft-api --bench special_bench
//!   optimized: rch exec -- cargo bench -p ft-api --bench special_bench

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn bench_special(c: &mut Criterion) {
    let n = 1usize << 20; // 1M elements

    c.bench_function("lgamma_1m", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_randn(vec![n], false).unwrap();
        b.iter(|| black_box(s.lgamma_tensor(black_box(x)).unwrap()));
    });

    c.bench_function("digamma_1m", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_randn(vec![n], false).unwrap();
        b.iter(|| black_box(s.digamma_tensor(black_box(x)).unwrap()));
    });
}

criterion_group!(benches, bench_special);
criterion_main!(benches);
