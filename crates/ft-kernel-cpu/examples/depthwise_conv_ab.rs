//! A/B for the direct depthwise conv2d kernel vs the grouped-conv anti-pattern
//! (one im2col + sgemm_bt per channel, as functional_conv2d_grouped did before
//! the no-grad fast path). The per-group baseline here is CONSERVATIVE — the real
//! old path also built C narrow + a C-way cat (tape nodes), so the shipped win is
//! larger. Both compute the same depthwise convolution.
//!   cargo run -q --release -p ft-kernel-cpu --example depthwise_conv_ab
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{conv2d_forward_f64, depthwise_conv2d_forward_f64};
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
fn per_group_im2col(
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
) -> Vec<f64> {
    // C independent single-channel convs (in_ch=1,out_ch=1), like the grouped loop.
    let mut out = vec![0.0f64; n * c * oh * ow];
    for ci in 0..c {
        // gather this channel's padded planes for all n: [n, 1, ph, pw]
        let mut chan = vec![0.0f64; n * ph * pw];
        for ni in 0..n {
            let src = (ni * c + ci) * ph * pw;
            chan[ni * ph * pw..(ni + 1) * ph * pw].copy_from_slice(&padded[src..src + ph * pw]);
        }
        let w = &weight[ci * kh * kw..(ci + 1) * kh * kw];
        let res = conv2d_forward_f64(&chan, w, None, n, 1, ph, pw, kh, kw, oh, ow, sh, sw, 1);
        for ni in 0..n {
            let dst = (ni * c + ci) * oh * ow;
            out[dst..dst + oh * ow].copy_from_slice(&res[ni * oh * ow..(ni + 1) * oh * ow]);
        }
    }
    out
}

fn main() {
    // MobileNet-ish depthwise: [N, C, H, W], 3x3 stride 1, no pad (padded already).
    for &(n, c, ph, pw, kh, kw) in &[(16usize, 64usize, 58, 58, 3, 3), (8, 256, 30, 30, 3, 3)] {
        let (sh, sw) = (1usize, 1usize);
        let oh = (ph - kh) / sh + 1;
        let ow = (pw - kw) / sw + 1;
        let padded: Vec<f64> = (0..n * c * ph * pw)
            .map(|i| (i % 977) as f64 * 0.01)
            .collect();
        let weight: Vec<f64> = (0..c * kh * kw)
            .map(|i| (i % 51) as f64 * 0.1 - 2.0)
            .collect();
        let _ = TensorMeta::from_shape(vec![n, c, ph, pw], DType::F64, Device::Cpu);

        let direct = depthwise_conv2d_forward_f64(
            &padded, &weight, None, n, c, ph, pw, kh, kw, oh, ow, sh, sw,
        );
        let group = per_group_im2col(&padded, &weight, n, c, ph, pw, kh, kw, oh, ow, sh, sw);
        let bx = direct
            .iter()
            .zip(group.iter())
            .all(|(a, b)| (a - b).abs() < 1e-9);
        eprintln!("[depthwise {n}x{c}x{ph}x{pw} k{kh}x{kw}] within-1e-9: {bx}");
        assert!(bx, "direct depthwise diverged from per-group im2col");

        let reps = 30;
        let mut bo = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            std::hint::black_box(per_group_im2col(
                &padded, &weight, n, c, ph, pw, kh, kw, oh, ow, sh, sw,
            ));
            bo = bo.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let mut bn = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            std::hint::black_box(depthwise_conv2d_forward_f64(
                &padded, &weight, None, n, c, ph, pw, kh, kw, oh, ow, sh, sw,
            ));
            bn = bn.min(t.elapsed().as_secs_f64() * 1e3);
        }
        eprintln!(
            "  per-group-im2col {bo:.2} ms / direct {bn:.2} ms / speedup {:.2}x\n",
            bo / bn
        );
    }
}
