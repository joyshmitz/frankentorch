//! A/B + bit-exact for lp_pool2d no-grad forward parallelization (powf-per-window = compute-bound).
//! Replicates the OLD serial 4-deep loop for an honest before/after; asserts the production
//! (parallel) output equals it BIT-FOR-BIT.
//!   cargo run -q --release -p ft-api --example lp_pool2d_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
fn serial(
    inp: &[f64],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    p: f64,
) -> Vec<f64> {
    let h_out = (h - kh) / sh + 1;
    let w_out = (w - kw) / sw + 1;
    let mut out = vec![0.0; n * c * h_out * w_out];
    for b in 0..n {
        for ch in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut s = 0.0;
                    for ih in oh * sh..(oh * sh + kh).min(h) {
                        for iw in ow * sw..(ow * sw + kw).min(w) {
                            s += inp[b * c * h * w + ch * h * w + ih * w + iw].abs().powf(p);
                        }
                    }
                    out[b * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow] =
                        s.powf(1.0 / p);
                }
            }
        }
    }
    out
}

fn main() {
    let nthreads = rayon::current_num_threads();
    let (n, c, h, w) = (16usize, 64, 64, 64);
    let (kh, kw, sh, sw, p) = (2usize, 2, 2, 2, 3.0);
    let data: Vec<f64> = (0..n * c * h * w)
        .map(|i| ((i % 197) as f64) * 0.01 - 1.0)
        .collect();
    let want = serial(&data, n, c, h, w, kh, kw, sh, sw, p);

    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    let new = pn.install(|| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let v = s
            .tensor_variable(data.clone(), vec![n, c, h, w], false)
            .unwrap();
        let o = s
            .functional_lp_pool2d(v, p, (kh, kw), Some((sh, sw)), false)
            .unwrap();
        let got = s.tensor_values(o).unwrap();
        assert_eq!(got.len(), want.len());
        for (g, wv) in got.iter().zip(want.iter()) {
            assert_eq!(g.to_bits(), wv.to_bits(), "lp_pool2d parallel != serial");
        }
        let mut best = f64::INFINITY;
        for _ in 0..15 {
            let v = s
                .tensor_variable(data.clone(), vec![n, c, h, w], false)
                .unwrap();
            let t = Instant::now();
            std::hint::black_box(
                s.functional_lp_pool2d(v, p, (kh, kw), Some((sh, sw)), false)
                    .unwrap(),
            );
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    });
    let mut old = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        std::hint::black_box(serial(&data, n, c, h, w, kh, kw, sh, sw, p));
        old = old.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "lp_pool2d [{n},{c},{h},{w}] k{kh} p{p} (bit-exact OK): OLD serial {old:.2}ms  NEW({nthreads}t) {new:.2}ms  =>  {:.2}x",
        old / new
    );
}
