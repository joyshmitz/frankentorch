//! A/B for the direct depthwise conv2d BACKWARD kernel vs the grouped
//! anti-pattern (one conv2d_backward per channel, as the grad path did before).
//! Conservative baseline (real old path also builds C narrows + a cat in the
//! tape). Both compute the same depthwise gradients.
//!   cargo run -q --release -p ft-kernel-cpu --example depthwise_conv_backward_ab
use ft_kernel_cpu::{conv2d_backward_f64, depthwise_conv2d_backward_f64};
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
fn per_group(
    dout: &[f64],
    padded: &[f64],
    weight: &[f64],
    n: usize,
    c: usize,
    ph: usize,
    pw: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
) -> (Vec<f64>, Vec<f64>) {
    let inplane = ph * pw;
    let outplane = oh * ow;
    let wsz = kh * kw;
    let mut dpadded = vec![0.0f64; n * c * inplane];
    let mut dweight = vec![0.0f64; c * wsz];
    for ci in 0..c {
        let mut dch = vec![0.0f64; n * outplane];
        let mut pch = vec![0.0f64; n * inplane];
        for ni in 0..n {
            dch[ni * outplane..(ni + 1) * outplane].copy_from_slice(
                &dout[(ni * c + ci) * outplane..(ni * c + ci) * outplane + outplane],
            );
            pch[ni * inplane..(ni + 1) * inplane].copy_from_slice(
                &padded[(ni * c + ci) * inplane..(ni * c + ci) * inplane + inplane],
            );
        }
        let w = &weight[ci * wsz..(ci + 1) * wsz];
        let (dp, dw, _) = conv2d_backward_f64(
            &dch, &pch, w, n, 1, ph, pw, kh, kw, oh, ow, sh, sw, 1, false,
        );
        for ni in 0..n {
            dpadded[(ni * c + ci) * inplane..(ni * c + ci) * inplane + inplane]
                .copy_from_slice(&dp[ni * inplane..(ni + 1) * inplane]);
        }
        dweight[ci * wsz..(ci + 1) * wsz].copy_from_slice(&dw);
    }
    (dpadded, dweight)
}

fn main() {
    for &(n, c, ph, pw) in &[(16usize, 64usize, 58, 58), (8, 256, 30, 30)] {
        let (kh, kw, sh, sw) = (3usize, 3, 1, 1);
        let oh = (ph - kh) / sh + 1;
        let ow = (pw - kw) / sw + 1;
        let padded: Vec<f64> = (0..n * c * ph * pw)
            .map(|i| (i % 877) as f64 * 0.01)
            .collect();
        let weight: Vec<f64> = (0..c * kh * kw)
            .map(|i| (i % 47) as f64 * 0.1 - 2.0)
            .collect();
        let dout: Vec<f64> = (0..n * c * oh * ow)
            .map(|i| (i % 53) as f64 * 0.05 - 1.3)
            .collect();

        let (dp_d, dw_d, _) = depthwise_conv2d_backward_f64(
            &dout, &padded, &weight, n, c, ph, pw, kh, kw, oh, ow, sh, sw, false,
        );
        let (dp_g, dw_g) = per_group(
            &dout, &padded, &weight, n, c, ph, pw, kh, kw, oh, ow, sh, sw,
        );
        let bx = dp_d
            .iter()
            .zip(dp_g.iter())
            .all(|(a, b)| (a - b).abs() < 1e-9)
            && dw_d
                .iter()
                .zip(dw_g.iter())
                .all(|(a, b)| (a - b).abs() < 1e-9);
        eprintln!("[dw-bwd {n}x{c}x{ph}x{pw} k3x3] within-1e-9: {bx}");
        assert!(bx, "direct depthwise backward diverged from per-group");

        let reps = 30;
        let mut bo = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            std::hint::black_box(per_group(
                &dout, &padded, &weight, n, c, ph, pw, kh, kw, oh, ow, sh, sw,
            ));
            bo = bo.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let mut bn = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            std::hint::black_box(depthwise_conv2d_backward_f64(
                &dout, &padded, &weight, n, c, ph, pw, kh, kw, oh, ow, sh, sw, false,
            ));
            bn = bn.min(t.elapsed().as_secs_f64() * 1e3);
        }
        eprintln!(
            "  per-group {bo:.2} ms / direct {bn:.2} ms / speedup {:.2}x\n",
            bo / bn
        );
    }
}
