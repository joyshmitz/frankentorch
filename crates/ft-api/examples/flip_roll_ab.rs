//! A/B + correctness for kgs4.104: flip_slice / roll_slice (ft-autograd, all dtypes)
//! gathered serially; now parallel. Verifies parallel == serial reference BIT-FOR-BIT
//! at large size, then times old-serial vs new.
//!   cargo run -q --release -p ft-api --example flip_roll_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}
fn flip_serial(v: &[f64], shape: &[usize], dims: &[usize]) -> Vec<f64> {
    let n: usize = shape.iter().product();
    let st = strides(shape);
    let mut out = vec![0.0; n];
    for (f, val) in v.iter().enumerate() {
        let mut rem = f;
        let mut dst = 0;
        for d in 0..shape.len() {
            let c = rem / st[d];
            rem %= st[d];
            let c2 = if dims.contains(&d) {
                shape[d] - 1 - c
            } else {
                c
            };
            dst += c2 * st[d];
        }
        out[dst] = *val;
    }
    out
}

fn main() {
    let nthreads = rayon::current_num_threads();
    let shape = [2048usize, 2048];
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n).map(|i| (i % 1000) as f64 * 0.001).collect();
    let want = flip_serial(&data, &shape, &[0, 1]);

    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    let new = pn.install(|| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let v = s
            .tensor_variable(data.clone(), shape.to_vec(), false)
            .unwrap();
        let f = s.tensor_flip(v, &[0, 1]).unwrap();
        let got = s.tensor_values(f).unwrap();
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.to_bits(), w.to_bits(), "flip parallel != serial");
        }
        let mut best = f64::INFINITY;
        for _ in 0..15 {
            let v = s
                .tensor_variable(data.clone(), shape.to_vec(), false)
                .unwrap();
            let t = Instant::now();
            std::hint::black_box(s.tensor_flip(v, &[0, 1]).unwrap());
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    });
    let mut old = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        std::hint::black_box(flip_serial(&data, &shape, &[0, 1]));
        old = old.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "flip [2048,2048] (bit-exact OK): OLD serial {old:.2}ms  NEW({nthreads}t) {new:.2}ms  =>  {:.2}x",
        old / new
    );
}
