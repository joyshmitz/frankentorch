use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ft_core::{DType, Device, ScalarTensor};
use ft_device::{DeviceError, DeviceGuard, ensure_same_device};
use std::fmt::Write as _;

fn bench_device_guard(c: &mut Criterion) {
    let mut group = c.benchmark_group("device_guard");
    let cpu_tensor = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
    let rhs_tensor = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
    let guard = DeviceGuard::new(Device::Cpu);

    group.bench_function("ensure_tensor_device_match_65536", |b| {
        b.iter(|| {
            let mut accepted = 0usize;
            for _ in 0..65_536 {
                if black_box(&guard)
                    .ensure_tensor_device(black_box(&cpu_tensor))
                    .is_ok()
                {
                    accepted = accepted.wrapping_add(1);
                }
            }
            black_box(accepted)
        });
    });

    group.bench_function("ensure_same_device_match_65536", |b| {
        b.iter(|| {
            let mut accepted = 0usize;
            for _ in 0..65_536 {
                if ensure_same_device(black_box(&cpu_tensor), black_box(&rhs_tensor)).is_ok() {
                    accepted = accepted.wrapping_add(1);
                }
            }
            black_box(accepted)
        });
    });

    let mismatch = DeviceError::Mismatch {
        expected: Device::Cpu,
        actual: Device::Cuda,
    };
    group.bench_function("mismatch_display_65536", |b| {
        b.iter(|| {
            let mut total_len = 0usize;
            let mut message = String::with_capacity(64);
            for _ in 0..65_536 {
                message.clear();
                if write!(&mut message, "{}", black_box(mismatch)).is_ok() {
                    total_len = total_len.wrapping_add(message.len());
                }
            }
            black_box(total_len)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_device_guard);
criterion_main!(benches);
