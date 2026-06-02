use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ft_core::{DType, Device, TensorMeta};

fn bench_tensor_meta(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_meta");
    let meta = TensorMeta::from_shape(vec![2, 3, 5, 7, 11, 13, 17, 19], DType::F64, Device::Cpu);

    group.bench_function("numel_rank8_repeated_65536", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for _ in 0..65_536 {
                total = total.wrapping_add(black_box(&meta).numel());
            }
            black_box(total)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_tensor_meta);
criterion_main!(benches);
