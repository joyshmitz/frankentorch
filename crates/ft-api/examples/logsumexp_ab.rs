//! Inference (no-grad) before/after for logsumexp(dim) forward.
//! OLD: serial per-lane max+sum-exp reduction + the two save_for_backward clones the
//! forward closure used to run unconditionally (vals.to_vec() + result.clone()).
//! NEW: production no-grad path — output lanes mapped over Rayon AND the backward-only
//! clones skipped. Asserts the production output equals the serial reference BIT-FOR-BIT.
//!   cargo run -q --release -p ft-api --example logsumexp_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_serial_with_saves(vals: &[f64], outer: usize, dim: usize, inner: usize) -> Vec<f64> {
    let out_len = outer * inner;
    let mut result = Vec::with_capacity(out_len);
    for o in 0..outer {
        for i in 0..inner {
            let mut max_val = f64::NEG_INFINITY;
            for d in 0..dim {
                let idx = o * dim * inner + d * inner + i;
                if vals[idx] > max_val {
                    max_val = vals[idx];
                }
            }
            if max_val.is_infinite() {
                result.push(max_val);
                continue;
            }
            let mut sum_exp = 0.0f64;
            for d in 0..dim {
                let idx = o * dim * inner + d * inner + i;
                sum_exp += (vals[idx] - max_val).exp();
            }
            result.push(max_val + sum_exp.ln());
        }
    }
    let _x = std::hint::black_box(vals.to_vec());
    let _y = std::hint::black_box(result.clone());
    result
}

fn main() {
    let nthreads = rayon::current_num_threads();
    // reduce dim=1 of [rows, cols] -> [rows]. outer=rows, dim=cols, inner=1.
    let (rows, cols) = (16384usize, 256);
    let n = rows * cols;
    let data: Vec<f64> = (0..n).map(|i| ((i % 211) as f64) * 0.01 - 1.0).collect();
    let want = old_serial_with_saves(&data, rows, cols, 1);

    let mut old = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        std::hint::black_box(old_serial_with_saves(&data, rows, cols, 1));
        old = old.min(t.elapsed().as_secs_f64() * 1e3);
    }

    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    let new = pn.install(|| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let v = s
            .tensor_variable(data.clone(), vec![rows, cols], false)
            .unwrap();
        let r = s.tensor_logsumexp(v, 1).unwrap();
        let got = s.tensor_values(r).unwrap();
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.to_bits(), w.to_bits(), "logsumexp parallel != serial");
        }
        let mut best = f64::INFINITY;
        for _ in 0..15 {
            let v = s
                .tensor_variable(data.clone(), vec![rows, cols], false)
                .unwrap();
            let t = Instant::now();
            let r = s.tensor_logsumexp(v, 1).unwrap();
            std::hint::black_box(r);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    });
    println!(
        "logsumexp no-grad [{rows},{cols}] dim=1 (bit-exact OK): OLD serial+saves {old:.2}ms  NEW({nthreads}t) {new:.2}ms  =>  {:.2}x",
        old / new
    );
}
