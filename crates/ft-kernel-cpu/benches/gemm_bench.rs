//! GEMM throughput bench. Run single-threaded vs multi-threaded with the same
//! binary by toggling `RAYON_NUM_THREADS`:
//!   baseline:  RAYON_NUM_THREADS=1 cargo bench -p ft-kernel-cpu --bench gemm_bench
//!   optimized: cargo bench -p ft-kernel-cpu --bench gemm_bench
//! The matmul kernel only splits across rows when more than one rayon thread is
//! available, so RAYON_NUM_THREADS=1 exercises the original single-call path.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{
    matmul_tensor_contiguous_f32, matmul_tensor_contiguous_f32_into, matmul_tensor_contiguous_f64,
};

fn bench_gemm(c: &mut Criterion) {
    for &(m, k, n) in &[(512usize, 512usize, 512usize), (1024, 1024, 1024)] {
        let lhs: Vec<f64> = (0..m * k).map(|i| (i % 100) as f64 * 0.01).collect();
        let rhs: Vec<f64> = (0..k * n).map(|i| (i % 100) as f64 * 0.02).collect();
        let lm = TensorMeta::from_shape(vec![m, k], DType::F64, Device::Cpu);
        let rm = TensorMeta::from_shape(vec![k, n], DType::F64, Device::Cpu);
        c.bench_function(&format!("matmul_f64_{m}x{k}x{n}"), |b| {
            b.iter(|| {
                black_box(
                    matmul_tensor_contiguous_f64(black_box(&lhs), black_box(&rhs), &lm, &rm)
                        .unwrap(),
                )
            })
        });

        let lhs32: Vec<f32> = (0..m * k).map(|i| (i % 100) as f32 * 0.01).collect();
        let rhs32: Vec<f32> = (0..k * n).map(|i| (i % 100) as f32 * 0.02).collect();
        let lm32 = TensorMeta::from_shape(vec![m, k], DType::F32, Device::Cpu);
        let rm32 = TensorMeta::from_shape(vec![k, n], DType::F32, Device::Cpu);
        c.bench_function(&format!("matmul_f32_{m}x{k}x{n}"), |b| {
            b.iter(|| {
                black_box(
                    matmul_tensor_contiguous_f32(
                        black_box(&lhs32),
                        black_box(&rhs32),
                        &lm32,
                        &rm32,
                    )
                    .unwrap(),
                )
            })
        });
    }
}

// f32 GEMM shapes drawn from a whisper-large encoder window (M = 1500 frames,
// embed = 1280). These are the consumer (`franken_whisper`) hot path: each
// encoder block runs attention qkv/out projections ([1500,1280]x[1280,1280]),
// an MLP up-projection ([1500,1280]x[1280,5120]) and an MLP down-projection
// ([1500,5120]x[5120,1280]). All three are TALL (m >> the col gate's 4*m vs n
// ratio is not met), so they take the row-split parallel path, not the column
// path — these benches exist to tune that path for these shapes.
fn bench_gemm_whisper_f32(c: &mut Criterion) {
    for &(m, k, n) in &[
        (1500usize, 1280usize, 1280usize), // attn qkv / out projection
        (1500, 1280, 5120),                // mlp fc (up)
        (1500, 5120, 1280),                // mlp proj (down)
    ] {
        let lhs: Vec<f32> = (0..m * k).map(|i| (i % 100) as f32 * 0.01).collect();
        let rhs: Vec<f32> = (0..k * n).map(|i| (i % 100) as f32 * 0.02).collect();
        let lm = TensorMeta::from_shape(vec![m, k], DType::F32, Device::Cpu);
        let rm = TensorMeta::from_shape(vec![k, n], DType::F32, Device::Cpu);
        c.bench_function(&format!("matmul_whisper_f32_{m}x{k}x{n}"), |b| {
            b.iter(|| {
                black_box(
                    matmul_tensor_contiguous_f32(black_box(&lhs), black_box(&rhs), &lm, &rm)
                        .unwrap(),
                )
            })
        });
    }
}

// Same whisper-large shapes, but through the buffer-reusing `_into` API with a
// warm, already-sized scratch buffer — the way the consumer's encoder drives it
// (one scratch per shape, reused across the 32 encoder layers). The delta vs
// `matmul_whisper_f32_*` isolates the per-call allocation + zero-init cost.
fn bench_gemm_whisper_f32_into(c: &mut Criterion) {
    for &(m, k, n) in &[
        (1500usize, 1280usize, 1280usize),
        (1500, 1280, 5120),
        (1500, 5120, 1280),
    ] {
        let lhs: Vec<f32> = (0..m * k).map(|i| (i % 100) as f32 * 0.01).collect();
        let rhs: Vec<f32> = (0..k * n).map(|i| (i % 100) as f32 * 0.02).collect();
        let lm = TensorMeta::from_shape(vec![m, k], DType::F32, Device::Cpu);
        let rm = TensorMeta::from_shape(vec![k, n], DType::F32, Device::Cpu);
        // Pre-warm the scratch to the output size so the steady-state cost
        // (resident pages, no realloc/memset) is what gets measured.
        let mut out = vec![0.0f32; m * n];
        c.bench_function(&format!("matmul_whisper_f32_into_{m}x{k}x{n}"), |b| {
            b.iter(|| {
                matmul_tensor_contiguous_f32_into(
                    &mut out,
                    black_box(&lhs),
                    black_box(&rhs),
                    &lm,
                    &rm,
                )
                .unwrap();
                black_box(&out);
            })
        });
    }
}

criterion_group!(
    benches,
    bench_gemm,
    bench_gemm_whisper_f32,
    bench_gemm_whisper_f32_into
);
criterion_main!(benches);
