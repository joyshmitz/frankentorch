//! A/B + correctness for kgs4.103: repeat_slice (ft-autograd, all dtypes) gathered its
//! output with a serial per-element index-unravel; now parallel. Verifies the parallel
//! output equals an inline serial reference BIT-FOR-BIT, then times old-serial vs new.
//!   cargo run -q --release -p ft-api --example repeat_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

// the pre-commit serial fill (reference), generic over the repeat formula.
fn serial(values: &[f64], in_shape: &[usize], repeats: &[usize]) -> Vec<f64> {
    let out_shape: Vec<usize> = in_shape.iter().zip(repeats).map(|(s, r)| s * r).collect();
    let out_numel: usize = out_shape.iter().product();
    let ostr = strides(&out_shape);
    let sstr = strides(in_shape);
    let mut out = vec![0.0; out_numel];
    for (flat, slot) in out.iter_mut().enumerate() {
        let mut rem = flat;
        let mut src = 0;
        for d in 0..repeats.len() {
            let c = rem / ostr[d];
            rem %= ostr[d];
            src += (c % in_shape[d]) * sstr[d];
        }
        *slot = values[src];
    }
    out
}
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

fn main() {
    let nthreads = rayon::current_num_threads();
    let in_shape = [256usize, 256];
    let repeats = [8usize, 8]; // out [2048, 2048] = 4.2M
    let data: Vec<f64> = (0..in_shape[0] * in_shape[1])
        .map(|i| (i % 1000) as f64 * 0.001)
        .collect();
    let want = serial(&data, &in_shape, &repeats);

    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    let new = pn.install(|| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let v = s
            .tensor_variable(data.clone(), in_shape.to_vec(), false)
            .unwrap();
        let r = s.tensor_repeat(v, &repeats).unwrap();
        let got = s.tensor_values(r).unwrap();
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.to_bits(), w.to_bits(), "repeat parallel != serial");
        }
        let mut best = f64::INFINITY;
        for _ in 0..15 {
            let v = s
                .tensor_variable(data.clone(), in_shape.to_vec(), false)
                .unwrap();
            let t = Instant::now();
            std::hint::black_box(s.tensor_repeat(v, &repeats).unwrap());
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    });
    let mut old = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        std::hint::black_box(serial(&data, &in_shape, &repeats));
        old = old.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "repeat [256,256]x[8,8] -> 4.2M (bit-exact OK): OLD serial {old:.2}ms  NEW({nthreads}t) {new:.2}ms  =>  {:.2}x",
        old / new
    );
}
