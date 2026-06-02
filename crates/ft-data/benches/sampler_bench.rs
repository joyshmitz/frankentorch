use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use ft_data::RandomSampler;

fn bench_random_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampler");

    group.bench_function("without_replacement_repeated_passes_4096x256", |b| {
        b.iter_batched(
            || {
                RandomSampler::new(4096)
                    .with_num_samples(4096 * 256 + 1537)
                    .with_seed(0xA5A5_5A5A_0123_4567)
            },
            |sampler| black_box(sampler.indices()),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_random_sampler);
criterion_main!(benches);
