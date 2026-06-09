use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use ft_api::FrankenTorchSession;
use ft_autograd::{TensorBackwardReport, TensorNodeId};
use ft_core::ExecutionMode;
use ft_optim::{Adam, AdamW, Optimizer, SGD};
use rayon::prelude::*;

// ── Cross-parameter parallelism probe (same-worker A/B) ────────────────────
// Measures the realistic ceiling of parallelizing the AdamW step ACROSS the 64
// independent parameters (a different parallelization model than the current
// per-param inner loop, which is below the rayon threshold and always serial).
// All three variants run in ONE process over scoped rayon pools so the
// serial/1-thread/N-thread comparison is a genuine same-worker A/B — not the
// separate-`rch exec` worker-variance trap. Operates on owned buffers (no
// session/tape) so it isolates the compute ceiling from copy/plumbing cost.
struct ProbeState {
    params: Vec<Vec<f64>>,
    grads: Vec<Vec<f64>>,
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
}

fn make_probe_state() -> ProbeState {
    let n_params = 64usize;
    let len = 1024usize;
    let params = (0..n_params)
        .map(|p| (0..len).map(|i| 0.001 * (p * len + i) as f64).collect())
        .collect();
    let grads = (0..n_params)
        .map(|p| {
            (0..len)
                .map(|i| 0.0005 * ((p * len + i) % 97) as f64)
                .collect()
        })
        .collect();
    let m = vec![vec![0.0f64; len]; n_params];
    let v = vec![vec![0.0f64; len]; n_params];
    ProbeState {
        params,
        grads,
        m,
        v,
    }
}

// One parameter's full AdamW update (non-amsgrad, weight_decay=0.01) — the exact
// per-element arithmetic and order used by AdamW::step.
#[inline]
fn adamw_one_param(p: &mut [f64], g: &[f64], m: &mut [f64], v: &mut [f64]) {
    let beta1 = 0.9;
    let beta2 = 0.999;
    let lr = 0.001;
    let eps = 1e-8;
    let weight_decay = 0.01;
    let bias_correction1 = 1.0 - beta1; // t=1
    let bias_correction2 = 1.0 - beta2;
    for (((p, &g), m_val), v_val) in p
        .iter_mut()
        .zip(g.iter())
        .zip(m.iter_mut())
        .zip(v.iter_mut())
    {
        *m_val = beta1 * *m_val + (1.0 - beta1) * g;
        *v_val = beta2 * *v_val + (1.0 - beta2) * g * g;
        let m_hat = *m_val / bias_correction1;
        let v_hat = *v_val / bias_correction2;
        let adam_delta = lr * m_hat / (v_hat.sqrt() + eps);
        let decay_delta = *p * lr * weight_decay;
        *p -= decay_delta + adam_delta;
    }
}

fn run_probe_serial(s: &mut ProbeState) {
    for (((p, g), m), v) in s
        .params
        .iter_mut()
        .zip(s.grads.iter())
        .zip(s.m.iter_mut())
        .zip(s.v.iter_mut())
    {
        adamw_one_param(p, g, m, v);
    }
}

fn run_probe_parallel(s: &mut ProbeState) {
    s.params
        .par_iter_mut()
        .zip(s.grads.par_iter())
        .zip(s.m.par_iter_mut())
        .zip(s.v.par_iter_mut())
        .for_each(|(((p, g), m), v)| adamw_one_param(p, g, m, v));
}

fn bench_crossparam_probe(c: &mut Criterion) {
    let mut group = c.benchmark_group("adamw_probe");

    group.bench_function("serial", |b| {
        b.iter_batched(
            make_probe_state,
            |mut s| {
                run_probe_serial(&mut s);
                black_box(s.params[0][0])
            },
            BatchSize::SmallInput,
        );
    });

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("pool1");
    group.bench_function("parallel_1thread", |b| {
        b.iter_batched(
            make_probe_state,
            |mut s| {
                pool1.install(|| run_probe_parallel(&mut s));
                black_box(s.params[0][0])
            },
            BatchSize::SmallInput,
        );
    });

    for threads in [4usize, 8, 16, 32] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("poolN");
        group.bench_function(format!("parallel_{threads}thread"), |b| {
            b.iter_batched(
                make_probe_state,
                |mut s| {
                    pool.install(|| run_probe_parallel(&mut s));
                    black_box(s.params[0][0])
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn make_adamw_case() -> (
    FrankenTorchSession,
    AdamW,
    TensorBackwardReport,
    TensorNodeId,
) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let mut params = Vec::with_capacity(64);
    let mut total_loss = None;

    for param_idx in 0..64 {
        let values = (0..1024)
            .map(|value_idx| 0.001 * (param_idx * 1024 + value_idx) as f64)
            .collect::<Vec<_>>();
        let param = session
            .tensor_variable(values, vec![1024], true)
            .expect("param");
        let param_loss = session.tensor_sum(param).expect("param sum");
        total_loss = Some(match total_loss {
            Some(acc) => session.tensor_add(acc, param_loss).expect("loss add"),
            None => param_loss,
        });
        params.push(param);
    }

    let first_param = params[0];
    let report = session
        .tensor_backward(total_loss.expect("total loss"))
        .expect("backward");
    let optimizer = AdamW::new(params, 0.001);
    (session, optimizer, report, first_param)
}

fn make_adam_case() -> (
    FrankenTorchSession,
    Adam,
    TensorBackwardReport,
    TensorNodeId,
) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let mut params = Vec::with_capacity(64);
    let mut total_loss = None;

    for param_idx in 0..64 {
        let values = (0..1024)
            .map(|value_idx| 0.001 * (param_idx * 1024 + value_idx) as f64)
            .collect::<Vec<_>>();
        let param = session
            .tensor_variable(values, vec![1024], true)
            .expect("param");
        let param_loss = session.tensor_sum(param).expect("param sum");
        total_loss = Some(match total_loss {
            Some(acc) => session.tensor_add(acc, param_loss).expect("loss add"),
            None => param_loss,
        });
        params.push(param);
    }

    let first_param = params[0];
    let report = session
        .tensor_backward(total_loss.expect("total loss"))
        .expect("backward");
    let optimizer = Adam::new(params, 0.001).weight_decay(0.01);
    (session, optimizer, report, first_param)
}

fn bench_adamw_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("adamw");

    group.bench_function("step_64x1024", |b| {
        b.iter_batched(
            make_adamw_case,
            |(mut session, mut optimizer, report, first_param)| {
                optimizer.step(&mut session, &report).expect("step");
                black_box(session.tensor_values(first_param).expect("values")[0])
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_adam_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("adam");

    group.bench_function("step_64x1024", |b| {
        b.iter_batched(
            make_adam_case,
            |(mut session, mut optimizer, report, first_param)| {
                optimizer.step(&mut session, &report).expect("step");
                black_box(session.tensor_values(first_param).expect("values")[0])
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn make_sgd_case() -> (FrankenTorchSession, SGD, TensorBackwardReport, TensorNodeId) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let mut params = Vec::with_capacity(64);
    let mut total_loss = None;

    for param_idx in 0..64 {
        let values = (0..1024)
            .map(|value_idx| 0.001 * (param_idx * 1024 + value_idx) as f64)
            .collect::<Vec<_>>();
        let param = session
            .tensor_variable(values, vec![1024], true)
            .expect("param");
        let param_loss = session.tensor_sum(param).expect("param sum");
        total_loss = Some(match total_loss {
            Some(acc) => session.tensor_add(acc, param_loss).expect("loss add"),
            None => param_loss,
        });
        params.push(param);
    }

    let first_param = params[0];
    let report = session
        .tensor_backward(total_loss.expect("total loss"))
        .expect("backward");
    let optimizer = SGD::new(params, 0.01).momentum(0.9).weight_decay(0.01);
    (session, optimizer, report, first_param)
}

fn bench_sgd_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("sgd");
    group.bench_function("step_64x1024", |b| {
        b.iter_batched(
            make_sgd_case,
            |(mut session, mut optimizer, report, first_param)| {
                optimizer.step(&mut session, &report).expect("step");
                black_box(session.tensor_values(first_param).expect("values")[0])
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_adamw_step,
    bench_adam_step,
    bench_sgd_step,
    bench_crossparam_probe
);
criterion_main!(benches);
