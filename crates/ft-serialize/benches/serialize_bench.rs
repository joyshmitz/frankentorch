use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ft_serialize::load_state_dict_from_bytes;

fn native_many_small_f64_payload(tensors: usize, width: usize) -> Vec<u8> {
    let key_bytes = "layer.0000.weight".len();
    let per_tensor_bytes = 8 + key_bytes + 8 + 8 + 1 + width * 8;
    let mut data = Vec::with_capacity(4 + 4 + 8 + tensors * per_tensor_bytes);
    data.extend_from_slice(b"FTSV");
    data.extend_from_slice(&1_u32.to_le_bytes());
    data.extend_from_slice(&(tensors as u64).to_le_bytes());

    for tensor_idx in 0..tensors {
        let key = format!("layer.{tensor_idx:04}.weight");
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&1_u64.to_le_bytes());
        data.extend_from_slice(&(width as u64).to_le_bytes());
        data.push(0);

        for value_idx in 0..width {
            let value = tensor_idx as f64 * 0.001 + value_idx as f64 * 0.000_001;
            data.extend_from_slice(&value.to_le_bytes());
        }
    }

    data
}

fn bench_native_load(c: &mut Criterion) {
    let payload = native_many_small_f64_payload(1024, 4);
    let mut group = c.benchmark_group("native_state_dict");
    group.bench_function("decode_many_small_f64_1024x4", |b| {
        b.iter(|| {
            let decoded = load_state_dict_from_bytes(black_box(&payload)).expect("decode");
            black_box(decoded.len())
        });
    });
    group.finish();
}

criterion_group!(benches, bench_native_load);
criterion_main!(benches);
