//! Same-process OLD full-panel vs NEW streaming conv3d_forward_f64 A/B + digest.
use ft_kernel_cpu::{
    conv3d_forward_f64, conv3d_im2col_f64, matmul_rhs_transposed_contiguous_f64_into,
};
use rayon::prelude::*;
use std::time::Instant;
fn fnv(v: &[f64]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}
fn t<F: FnMut()>(mut f: F, it: usize) -> f64 {
    f();
    let s = Instant::now();
    for _ in 0..it {
        f();
    }
    s.elapsed().as_secs_f64() * 1e3 / it as f64
}
#[allow(clippy::too_many_arguments)]
fn old(
    pd: &[f64],
    wt: &[f64],
    bias: &[f64],
    batch: usize,
    ic: usize,
    pdd: usize,
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
    oc: usize,
) -> Vec<f64> {
    let pwid = ic * kd * kh * kw;
    let pc = od * oh * ow;
    let flat = batch * pc;
    let panel = conv3d_im2col_f64(
        pd, batch, ic, pdd, ph, pw, kd, kh, kw, od, oh, ow, sd, sh, sw,
    );
    let mut of = vec![0.0f64; flat * oc];
    matmul_rhs_transposed_contiguous_f64_into(&mut of, flat, pwid, oc, &panel, wt).unwrap();
    let mut out = vec![0.0f64; batch * oc * pc];
    out.par_chunks_mut(pc).enumerate().for_each(|(idx, orow)| {
        let n = idx / oc;
        let o = idx % oc;
        let bo = bias[o];
        for p in 0..pc {
            orow[p] = of[(n * pc + p) * oc + o] + bo;
        }
    });
    out
}
fn main() {
    println!("threads={}", rayon::current_num_threads());
    let (batch, ic, oc, kd, kh, kw) = (2usize, 32usize, 32usize, 3usize, 3usize, 3usize);
    let (d, h, w) = (8usize, 16, 16);
    let (pdd, ph, pw) = (d + 2, h + 2, w + 2);
    let (od, oh, ow) = (d, h, w);
    let (sd, sh, sw) = (1usize, 1, 1);
    let pd: Vec<f64> = (0..batch * ic * pdd * ph * pw)
        .map(|i| (i % 251) as f64 * 0.001 - 0.12)
        .collect();
    let wt: Vec<f64> = (0..oc * ic * kd * kh * kw)
        .map(|i| (i % 97) as f64 * 0.002 - 0.1)
        .collect();
    let bias: Vec<f64> = (0..oc).map(|i| i as f64 * 0.01 - 0.3).collect();
    let it = 15;
    let dgo = fnv(&old(
        &pd, &wt, &bias, batch, ic, pdd, ph, pw, kd, kh, kw, od, oh, ow, sd, sh, sw, oc,
    ));
    let dgn = fnv(&conv3d_forward_f64(
        &pd,
        &wt,
        Some(&bias),
        batch,
        ic,
        pdd,
        ph,
        pw,
        kd,
        kh,
        kw,
        od,
        oh,
        ow,
        sd,
        sh,
        sw,
        oc,
    ));
    let mo = t(
        || {
            let _ = old(
                &pd, &wt, &bias, batch, ic, pdd, ph, pw, kd, kh, kw, od, oh, ow, sd, sh, sw, oc,
            );
        },
        it,
    );
    let mn = t(
        || {
            let _ = conv3d_forward_f64(
                &pd,
                &wt,
                Some(&bias),
                batch,
                ic,
                pdd,
                ph,
                pw,
                kd,
                kh,
                kw,
                od,
                oh,
                ow,
                sd,
                sh,
                sw,
                oc,
            );
        },
        it,
    );
    println!(
        "conv3d OLD={mo:.2}ms NEW={mn:.2}ms speedup={:.2}x digest_ok={}",
        mo / mn,
        dgo == dgn
    );
}
