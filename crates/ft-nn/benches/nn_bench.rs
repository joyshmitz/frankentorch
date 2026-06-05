use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use ft_nn::{GRU, LSTM, Module, MultiheadAttention};

fn make_mha_case(
    batch_size: usize,
    seq_len: usize,
    embed_dim: usize,
    num_heads: usize,
) -> (
    FrankenTorchSession,
    MultiheadAttention,
    ft_autograd::TensorNodeId,
) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let attention = MultiheadAttention::new(&mut session, embed_dim, num_heads).expect("mha");
    let values = (0..(batch_size * seq_len * embed_dim))
        .map(|idx| (idx % 251) as f64 * 0.001)
        .collect::<Vec<_>>();
    let input = session
        .tensor_variable(values, vec![batch_size, seq_len, embed_dim], true)
        .expect("input");
    (session, attention, input)
}

fn bench_multihead_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("multihead_attention");
    group.bench_function("forward_8x64x128_h8", |b| {
        b.iter_batched(
            || make_mha_case(8, 64, 128, 8),
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
    group.bench_function("forward_1x512x128_h8", |b| {
        b.iter_batched(
            || make_mha_case(1, 512, 128, 8),
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
            || make_mha_case(8, 64, 128, 8),
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
    group.bench_function("forward_1x512x128_h8_nograd", |b| {
        b.iter_batched(
            || make_mha_case(1, 512, 128, 8),
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
    group.bench_function("forward_1x1024x128_h8_nograd", |b| {
        b.iter_batched(
            || make_mha_case(1, 1024, 128, 8),
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

fn make_lstm_case(
    seq_len: usize,
    batch_size: usize,
    input_size: usize,
    hidden_size: usize,
) -> (FrankenTorchSession, LSTM, ft_autograd::TensorNodeId) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let lstm = LSTM::new(
        &mut session,
        input_size,
        hidden_size,
        1,
        false,
        0.0,
        false,
    )
    .expect("lstm");
    let values = (0..(seq_len * batch_size * input_size))
        .map(|idx| (idx % 251) as f64 * 0.001)
        .collect::<Vec<_>>();
    let input = session
        .tensor_variable(values, vec![seq_len, batch_size, input_size], true)
        .expect("input");
    (session, lstm, input)
}

fn bench_lstm_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("lstm");
    group.bench_function("forward_seq128_b16_in128_h256", |b| {
        b.iter_batched(
            || make_lstm_case(128, 16, 128, 256),
            |(mut session, lstm, input)| {
                let output = lstm.forward(&mut session, input).expect("forward");
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
    group.bench_function("forward_seq128_b16_in128_h256_nograd", |b| {
        b.iter_batched(
            || make_lstm_case(128, 16, 128, 256),
            |(mut session, lstm, input)| {
                session.no_grad_enter();
                let output = lstm.forward(&mut session, input).expect("forward");
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

fn make_gru_case(
    seq_len: usize,
    batch_size: usize,
    input_size: usize,
    hidden_size: usize,
) -> (FrankenTorchSession, GRU, ft_autograd::TensorNodeId) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let gru = GRU::new(
        &mut session,
        input_size,
        hidden_size,
        1,
        false,
        0.0,
        false,
    )
    .expect("gru");
    let values = (0..(seq_len * batch_size * input_size))
        .map(|idx| (idx % 251) as f64 * 0.001)
        .collect::<Vec<_>>();
    let input = session
        .tensor_variable(values, vec![seq_len, batch_size, input_size], true)
        .expect("input");
    (session, gru, input)
}

fn bench_gru_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("gru");
    group.bench_function("forward_seq128_b16_in128_h256", |b| {
        b.iter_batched(
            || make_gru_case(128, 16, 128, 256),
            |(mut session, gru, input)| {
                let output = gru.forward(&mut session, input).expect("forward");
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
    group.bench_function("forward_seq128_b16_in128_h256_nograd", |b| {
        b.iter_batched(
            || make_gru_case(128, 16, 128, 256),
            |(mut session, gru, input)| {
                session.no_grad_enter();
                let output = gru.forward(&mut session, input).expect("forward");
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
    bench_multihead_attention_nograd,
    bench_lstm_forward,
    bench_gru_forward
);
criterion_main!(benches);
