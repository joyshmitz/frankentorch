#![no_main]

use ft_core::{DType, DenseTensor, Device, TensorMeta, TensorStorage};
use ft_serialize::{load_safetensors_from_bytes, save_safetensors_to_bytes};
use libfuzzer_sys::fuzz_target;
use std::collections::BTreeMap;
use std::sync::Arc;

const MAX_INPUT_BYTES: usize = 512;
const MAX_TENSORS: usize = 4;
const MAX_NUMEL: usize = 64;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let n_tensors = usize::from(1 + (data[0] % MAX_TENSORS as u8)).min(MAX_TENSORS);
    let body = &data[1..];

    // Build a state_dict with n_tensors small F64 tensors.
    let mut state_dict: BTreeMap<String, DenseTensor> = BTreeMap::new();
    let mut cursor = 0usize;
    for t in 0..n_tensors {
        if cursor + 2 >= body.len() {
            return;
        }
        let rows = usize::from(1 + (body[cursor] % 8));
        let cols = usize::from(1 + (body[cursor + 1] % 8));
        let numel = rows * cols;
        if numel > MAX_NUMEL {
            return;
        }
        cursor += 2;
        let values: Vec<f64> = (0..numel)
            .map(|i| {
                let raw = body.get((cursor + i) % body.len().max(1)).copied().unwrap_or(0) as i32;
                (raw - 128) as f64 / 23.0
            })
            .collect();
        cursor += numel;
        let meta = TensorMeta::from_shape(vec![rows, cols], DType::F64, Device::Cpu);
        let storage = TensorStorage::F64(Arc::new(values));
        let tensor = match DenseTensor::from_typed_storage(meta, storage) {
            Ok(t) => t,
            Err(_) => return,
        };
        state_dict.insert(format!("tensor_{t}"), tensor);
    }

    // Roundtrip via in-memory save_safetensors_to_bytes +
    // load_safetensors_from_bytes (no file I/O).
    let bytes = match save_safetensors_to_bytes(&state_dict, None) {
        Ok(b) => b,
        Err(_) => return,
    };
    let loaded = match load_safetensors_from_bytes(&bytes) {
        Ok(m) => m,
        Err(e) => panic!("decode after encode failed: {e:?}"),
    };

    // Compare: each tensor's shape and values survive the round-trip.
    assert_eq!(loaded.len(), state_dict.len(), "tensor count mismatch");
    for (name, original) in &state_dict {
        let got = loaded
            .get(name)
            .unwrap_or_else(|| panic!("tensor {name} missing after roundtrip"));
        assert_eq!(
            got.meta().shape(),
            original.meta().shape(),
            "tensor {name} shape mismatch"
        );
        assert_eq!(
            got.meta().dtype(),
            original.meta().dtype(),
            "tensor {name} dtype mismatch"
        );
        // Compare values bit-exactly.
        let orig_storage = match original.typed_storage() {
            TensorStorage::F64(v) => v,
            _ => continue,
        };
        let got_storage = match got.typed_storage() {
            TensorStorage::F64(v) => v,
            _ => continue,
        };
        assert_eq!(
            orig_storage.len(),
            got_storage.len(),
            "tensor {name} value count mismatch"
        );
        for (i, (a, b)) in orig_storage.iter().zip(got_storage.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "tensor {name} value[{i}] bit mismatch: orig {} got {}",
                a,
                b
            );
        }
    }
});
