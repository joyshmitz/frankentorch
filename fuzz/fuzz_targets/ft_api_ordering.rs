#![no_main]

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_LEN: u8 = 32;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let n = usize::from(1 + (data[0] % MAX_LEN)); // 1..=32
    let k = usize::from(1 + (data[1] % n as u8).max(1)).min(n); // 1..=n
    let descending = data[2] & 1 == 1;
    let body = &data[3..];

    let input: Vec<f64> = (0..n)
        .map(|i| {
            let raw = body.get(i % body.len().max(1)).copied().unwrap_or(0) as i32;
            (raw - 128) as f64 / 17.0
        })
        .collect();

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = match s.tensor_variable(input.clone(), vec![n], false) {
        Ok(t) => t,
        Err(_) => return,
    };

    // --- sort ---
    if let Ok((sorted_t, _)) = s.tensor_sort(x, 0, descending) {
        if let Ok(sorted_vals) = s.tensor_values(sorted_t) {
            assert_eq!(sorted_vals.len(), n, "sort length");
            // Verify ordering on non-NaN values.
            let any_nan = input.iter().any(|v| v.is_nan());
            if !any_nan {
                for w in sorted_vals.windows(2) {
                    if descending {
                        assert!(
                            w[0] >= w[1],
                            "sort_descending broken: {} < {}",
                            w[0], w[1]
                        );
                    } else {
                        assert!(
                            w[0] <= w[1],
                            "sort_ascending broken: {} > {}",
                            w[0], w[1]
                        );
                    }
                }
            }

            // Permutation invariant: sorted_vals is a permutation of input.
            // Compare multisets via bit-counting (works for NaN too, modulo
            // NaN encoding).
            let mut input_bits: Vec<u64> = input.iter().map(|v| v.to_bits()).collect();
            let mut sorted_bits: Vec<u64> = sorted_vals.iter().map(|v| v.to_bits()).collect();
            input_bits.sort_unstable();
            sorted_bits.sort_unstable();
            assert_eq!(
                input_bits, sorted_bits,
                "sort must preserve the multiset"
            );
        }
    }

    // --- topk ---
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x2 = match s.tensor_variable(input.clone(), vec![n], false) {
        Ok(t) => t,
        Err(_) => return,
    };
    if let Ok((topk_t, _)) = s.tensor_topk(x2, k, 0, true, true) {
        if let Ok(topk_vals) = s.tensor_values(topk_t) {
            assert_eq!(topk_vals.len(), k, "topk length");
            // Largest=true, sorted=true: descending order.
            let any_nan = input.iter().any(|v| v.is_nan());
            if !any_nan {
                for w in topk_vals.windows(2) {
                    assert!(
                        w[0] >= w[1],
                        "topk(largest=true, sorted=true) not descending: {} < {}",
                        w[0], w[1]
                    );
                }
                // topk values are all in the input multiset.
                let input_bits: std::collections::HashSet<u64> =
                    input.iter().map(|v| v.to_bits()).collect();
                for &v in &topk_vals {
                    assert!(
                        input_bits.contains(&v.to_bits()),
                        "topk returned value {} not in input",
                        v
                    );
                }
            }
        }
    }

    // --- searchsorted ---
    // For searchsorted, sorted_sequence must be sorted. Build it by
    // running tensor_sort first.
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let seq_node = match s.tensor_variable(input.clone(), vec![n], false) {
        Ok(t) => t,
        Err(_) => return,
    };
    let sorted_seq = match s.tensor_sort(seq_node, 0, false) {
        Ok((t, _)) => t,
        Err(_) => return,
    };
    let needles: Vec<f64> = (0..k)
        .map(|i| {
            let raw = body.get((n + i) % body.len().max(1)).copied().unwrap_or(0) as i32;
            (raw - 128) as f64 / 17.0
        })
        .collect();
    let needles_node = match s.tensor_variable(needles.clone(), vec![k], false) {
        Ok(t) => t,
        Err(_) => return,
    };
    if let Ok(out_t) = s.tensor_searchsorted(sorted_seq, needles_node, false) {
        if let Ok(out_vals) = s.tensor_values(out_t) {
            assert_eq!(out_vals.len(), k, "searchsorted length");
            // Every returned index must be in [0, n] (inclusive — insertion
            // point at end is n).
            for (i, &v) in out_vals.iter().enumerate() {
                let idx = v as usize;
                assert!(
                    (v - idx as f64).abs() < f64::EPSILON,
                    "searchsorted idx[{i}] = {v} not integer"
                );
                assert!(
                    idx <= n,
                    "searchsorted idx[{i}] = {idx} > n = {n}"
                );
            }
        }
    }
});
