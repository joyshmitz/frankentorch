//! A/B for the direct depthwise conv3d kernel vs the grouped anti-pattern
//! (one im2col + GEMM per channel, as functional_conv3d_dilated did before the
//! no-grad fast path). Conservative baseline (real old path also builds C narrow
//! + a C-way cat). Both compute the same depthwise 3-D convolution.
//!   cargo run -q --release -p ft-kernel-cpu --example depthwise_conv3d_ab
use ft_kernel_cpu::{conv3d_forward_f64, depthwise_conv3d_forward_f64};
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
fn per_group(
    padded: &[f64],
    weight: &[f64],
    n: usize,
    c: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    od: usize,
    oh: usize,
    ow: usize,
    sd: usize,
    sh: usize,
    sw: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; n * c * od * oh * ow];
    let inplane = pd * ph * pw;
    let outplane = od * oh * ow;
    for ci in 0..c {
        let mut chan = vec![0.0f64; n * inplane];
        for ni in 0..n {
            let src = (ni * c + ci) * inplane;
            chan[ni * inplane..(ni + 1) * inplane].copy_from_slice(&padded[src..src + inplane]);
        }
        let w = &weight[ci * kd * kh * kw..(ci + 1) * kd * kh * kw];
        let res = conv3d_forward_f64(
            &chan, w, None, n, 1, pd, ph, pw, kd, kh, kw, od, oh, ow, sd, sh, sw, 1,
        );
        for ni in 0..n {
            let dst = (ni * c + ci) * outplane;
            out[dst..dst + outplane].copy_from_slice(&res[ni * outplane..(ni + 1) * outplane]);
        }
    }
    out
}

fn main() {
    // Video-ish depthwise conv3d: [N, C, D, H, W], 3x3x3, stride 1 (padded already).
    for &(n, c, pd, ph, pw) in &[(4usize, 64usize, 18, 18, 18), (2, 128, 16, 16, 16)] {
        let (kd, kh, kw, sd, sh, sw) = (3usize, 3, 3, 1, 1, 1);
        let od = (pd - kd) / sd + 1;
        let oh = (ph - kh) / sh + 1;
        let ow = (pw - kw) / sw + 1;
        let padded: Vec<f64> = (0..n * c * pd * ph * pw).map(|i| (i % 911) as f64 * 0.01).collect();
        let weight: Vec<f64> = (0..c * kd * kh * kw).map(|i| (i % 41) as f64 * 0.1 - 1.5).collect();

        let direct = depthwise_conv3d_forward_f64(
            &padded, &weight, None, n, c, pd, ph, pw, kd, kh, kw, od, oh, ow, sd, sh, sw,
        );
        let group = per_group(&padded, &weight, n, c, pd, ph, pw, kd, kh, kw, od, oh, ow, sd, sh, sw);
        let bx = direct.iter().zip(group.iter()).all(|(a, b)| (a - b).abs() < 1e-9);
        eprintln!("[depthwise3d {n}x{c}x{pd}x{ph}x{pw} k3x3x3] within-1e-9: {bx}");
        assert!(bx, "direct depthwise3d diverged from per-group");

        let reps = 20;
        let mut bo = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            std::hint::black_box(per_group(
                &padded, &weight, n, c, pd, ph, pw, kd, kh, kw, od, oh, ow, sd, sh, sw,
            ));
            bo = bo.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let mut bn = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            std::hint::black_box(depthwise_conv3d_forward_f64(
                &padded, &weight, None, n, c, pd, ph, pw, kd, kh, kw, od, oh, ow, sd, sh, sw,
            ));
            bn = bn.min(t.elapsed().as_secs_f64() * 1e3);
        }
        eprintln!("  per-group {bo:.2} ms / direct {bn:.2} ms / speedup {:.2}x\n", bo / bn);
    }
}
