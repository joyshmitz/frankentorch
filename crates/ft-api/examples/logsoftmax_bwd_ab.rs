//! Isolated lever A/B for the LogSoftmax BACKWARD contrib in ft-autograd. The
//! backward is `grad_i - exp(output_i)*sum_j(grad_j)` per (outer,inner) lane —
//! compute-bound on the per-element `exp`. Each `outer` lane is a contiguous
//! [reduce*inner] block, so it fans over par_chunks_mut. Times the serial triple
//! loop vs the parallel chunked loop, ONE process ONE worker, anchored. NEW
//! asserted BIT-FOR-BIT == serial.
//!   cargo run -q --release -p ft-api --example logsoftmax_bwd_ab
use rayon::prelude::*;
use std::time::Instant;

fn serial(incoming: &[f64], output: &[f64], outer: usize, reduce: usize, inner: usize) -> Vec<f64> {
    let mut out = vec![0.0; outer * reduce * inner];
    let stride = reduce * inner;
    for o in 0..outer {
        let base = o * stride;
        for i in 0..inner {
            let mut gs = 0.0;
            for r in 0..reduce {
                gs += incoming[base + r * inner + i];
            }
            for r in 0..reduce {
                let l = r * inner + i;
                out[base + l] = incoming[base + l] - output[base + l].exp() * gs;
            }
        }
    }
    out
}

fn par(incoming: &[f64], output: &[f64], outer: usize, reduce: usize, inner: usize) -> Vec<f64> {
    let stride = reduce * inner;
    let mut out = vec![0.0; outer * reduce * inner];
    out.par_chunks_mut(stride).enumerate().for_each(|(o, block)| {
        let base = o * stride;
        for i in 0..inner {
            let mut gs = 0.0;
            for r in 0..reduce {
                gs += incoming[base + r * inner + i];
            }
            for r in 0..reduce {
                let l = r * inner + i;
                block[l] = incoming[base + l] - output[base + l].exp() * gs;
            }
        }
    });
    out
}

fn bench(label: &str, outer: usize, reduce: usize, inner: usize) {
    let nthreads = rayon::current_num_threads();
    let n = outer * reduce * inner;
    let output: Vec<f64> = (0..n).map(|i| -((i % 997) as f64) * 0.005 - 0.5).collect();
    let incoming: Vec<f64> = (0..n).map(|i| 1.0 + ((i % 11) as f64) * 0.05).collect();
    let want = serial(&incoming, &output, outer, reduce, inner);
    let got = par(&incoming, &output, outer, reduce, inner);
    for (a, b) in want.iter().zip(got.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "parallel != serial");
    }
    let mut old = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        std::hint::black_box(serial(&incoming, &output, outer, reduce, inner));
        old = old.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut new = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        std::hint::black_box(par(&incoming, &output, outer, reduce, inner));
        new = new.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("{label} [{outer}x{reduce}x{inner}] ({nthreads}t, bit-exact OK): serial {old:.3}ms  par {new:.3}ms  =>  {:.2}x", old / new);
}

fn main() {
    bench("ANCHOR-small", 4, 64, 1);
    bench("logsoftmax-bwd", 4096, 256, 1);
    bench("logsoftmax-bwd", 4096, 1024, 1);
}
