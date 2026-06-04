use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use ft_nn::{Module, MultiheadAttention};

fn make_mha_case() -> (
    FrankenTorchSession,
    MultiheadAttention,
    ft_autograd::TensorNodeId,
) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let attention = MultiheadAttention::new(&mut session, 128, 8).expect("mha");
    let values = (0..(8 * 64 * 128))
        .map(|idx| (idx % 251) as f64 * 0.001)
        .collect::<Vec<_>>();
    let input = session
        .tensor_variable(values, vec![8, 64, 128], true)
        .expect("input");
    (session, attention, input)
}

fn bench_multihead_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("multihead_attention");
    group.bench_function("forward_8x64x128_h8", |b| {
        b.iter_batched(
            make_mha_case,
            |(mut session, attention, input)| {
                let output = attention.forward(&mut session, input).expect("forward");
                let first = session
                    .tensor_values(output)
                    .expect("values")
                    .first()
                    .copied()
                    .expect("nonempty output");
                black_box(first)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_multihead_attention_nograd(c: &mut Criterion) {
    let mut group = c.benchmark_group("multihead_attention");
    group.bench_function("forward_8x64x128_h8_nograd", |b| {
        b.iter_batched(
            make_mha_case,
            |(mut session, attention, input)| {
                session.no_grad_enter();
                let output = attention.forward(&mut session, input).expect("forward");
                let first = session
                    .tensor_values(output)
                    .expect("values")
                    .first()
                    .copied()
                    .expect("nonempty output");
                session.no_grad_exit();
                black_box(first)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_multihead_attention,
    bench_multihead_attention_nograd
);
criterion_main!(benches);
