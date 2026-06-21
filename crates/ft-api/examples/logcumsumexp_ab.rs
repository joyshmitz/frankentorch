//! Inference (no-grad) before/after for logcumsumexp forward.
//! OLD: serial outer/inner/dim scan + the two save_for_backward clones the forward
//! closure used to run unconditionally (vals.to_vec() + result.clone()).
//! NEW: production no-grad path — lanes fanned over Rayon AND the backward-only clones
//! skipped (apply_function returns a Leaf in no-grad, so they were dead work).
//! Asserts the production output equals the serial reference BIT-FOR-BIT.
//!   cargo run -q --release -p ft-api --example logcumsumexp_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_serial_with_saves(vals: &[f64], outer: usize, dim: usize, inner: usize) -> Vec<f64> {
    let mut result = vec![0.0f64; vals.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut running_max = f64::NEG_INFINITY;
            let mut running_sum = 0.0f64;
            for d in 0..dim {
                let idx = o * dim * inner + d * inner + i;
                let val = vals[idx];
                if val == running_max && val.is_infinite() {
                    result[idx] = val;
                    continue;
                }
                if val > running_max {
                    running_sum = running_sum * (running_max - val).exp() + 1.0;
                    running_max = val;
                } else {
                    running_sum += (val - running_max).exp();
                }
                result[idx] = running_max + running_sum.ln();
            }
        }
    }
    // the two save_for_backward clones the old forward closure ran unconditionally
    let _saved_x = std::hint::black_box(vals.to_vec());
    let _saved_y = std::hint::black_box(result.clone());
    result
}

fn main() {
    let nthreads = rayon::current_num_threads();
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
        let r = s.tensor_logcumsumexp(v, 1).unwrap();
        let got = s.tensor_values(r).unwrap();
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.to_bits(), w.to_bits(), "logcumsumexp parallel != serial");
        }
        let mut best = f64::INFINITY;
        for _ in 0..15 {
            let v = s
                .tensor_variable(data.clone(), vec![rows, cols], false)
                .unwrap();
            let t = Instant::now();
            let lcse = s.tensor_logcumsumexp(v, 1).unwrap();
            std::hint::black_box(lcse);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    });
    println!(
        "logcumsumexp no-grad [{rows},{cols}] dim=1 (bit-exact OK): OLD serial+saves {old:.2}ms  NEW({nthreads}t) {new:.2}ms  =>  {:.2}x",
        old / new
    );
}
