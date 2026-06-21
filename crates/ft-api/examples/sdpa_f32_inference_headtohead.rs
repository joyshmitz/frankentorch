//! f32 SDPA INFERENCE (no-grad forward) head-to-head vs PyTorch (cc).
//! f32 mirror of `sdpa_inference_headtohead` — the dominant production dtype for serving /
//! autoregressive decode. FT's no-grad f32 fast-path flash kernel (`sdpa_forward_f32`) vs
//! PyTorch f32 CPU SDPA. Non-causal + causal. Reports the RAW kernel and the full no-grad
//! API path; also prints the FT/PyTorch output-sum relative diff as a correctness check.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example sdpa_f32_inference_headtohead

use std::process::Command;
use std::time::Instant;

const BH: usize = 16;
const SEQ: usize = 512;
const D: usize = 64;

fn seq_f32(n: usize, shift: f32) -> Vec<f32> {
    (0..n).map(|i| (((i as f32) * 0.017 + shift).sin()) * 0.2).collect()
}

// NOTE: the no-grad f32 SDPA *session* path (`scaled_dot_product_attention` with
// requires_grad=false) currently returns `DenseTensor(UnsupportedDType(F32))` — a
// pre-existing API gap, filed separately. This example therefore benchmarks the RAW
// `sdpa_forward_f32` flash kernel (the thing the no-grad/grad f32 fast paths call) directly
// vs PyTorch's f32 SDPA, which is the kernel-level comparison this perf lever targets.
fn ft_kernel(qb: &[f32], kb: &[f32], vb: &[f32], causal: bool) -> (Vec<f32>, f64) {
    let scale = 1.0 / (D as f32).sqrt();
    let out = ft_kernel_cpu::sdpa_forward_f32(qb, kb, vb, BH, SEQ, SEQ, D, D, scale, causal);
    let sum = out.iter().map(|x| x.abs() as f64).sum();
    (out, sum)
}

const PY: &str = r#"
import os, time
import torch
import torch.nn.functional as F
BH, SEQ, D = 16, 512, 64
total = BH*SEQ*D
causal = os.environ.get("FT_CAUSAL","0") == "1"
def dv(shift):
    return (torch.arange(total, dtype=torch.float32).mul_(0.017).add_(shift).sin_().mul_(0.2).reshape(BH,SEQ,D))
iters = int(os.environ.get("FT_GAUNTLET_ITERS","20"))
torch.set_num_threads(int(os.environ.get("FT_TORCH_THREADS","8")))
q, k, v = dv(0.0), dv(1.0), dv(2.0)
with torch.no_grad():
    chk = 0.0
    for _ in range(3):
        chk = F.scaled_dot_product_attention(q,k,v,is_causal=causal).abs().sum().item()
    ts=[]
    for _ in range(iters):
        t0=time.perf_counter()
        F.scaled_dot_product_attention(q,k,v,is_causal=causal).abs().sum().item()
        ts.append((time.perf_counter()-t0)*1e3)
ts.sort()
print("ELAPSED_MS", ts[0])
print("CHECKSUM", chk)
"#;

fn bench_ft(qb: &[f32], kb: &[f32], vb: &[f32], causal: bool, iters: usize) -> (f64, f64) {
    for _ in 0..3 {
        let _ = ft_kernel(qb, kb, vb, causal);
    }
    let mut times = Vec::with_capacity(iters);
    let mut sum = 0.0;
    for _ in 0..iters {
        let t = Instant::now();
        let (o, s) = ft_kernel(qb, kb, vb, causal);
        std::hint::black_box(&o);
        sum = s;
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[0], sum)
}

fn py(causal: bool, iters: usize) -> Option<(f64, f64)> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let out = Command::new(&python)
        .arg("-c").arg(PY)
        .env("FT_GAUNTLET_ITERS", iters.to_string())
        .env("FT_CAUSAL", if causal { "1" } else { "0" })
        .output().ok()?;
    if !out.status.success() {
        eprintln!("pytorch failed: {}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let ms = s.lines().find_map(|l| l.strip_prefix("ELAPSED_MS ").and_then(|v| v.trim().parse().ok()))?;
    let chk = s.lines().find_map(|l| l.strip_prefix("CHECKSUM ").and_then(|v| v.trim().parse().ok()))?;
    Some((ms, chk))
}

fn report(qb: &[f32], kb: &[f32], vb: &[f32], label: &str, causal: bool, iters: usize) {
    let (ft, ft_sum) = bench_ft(qb, kb, vb, causal, iters);
    print!("  {label:12} FT {ft:8.3} ms");
    match py(causal, iters) {
        Some((p, ps)) => {
            let rel = (ft_sum - ps).abs() / (ps.abs() + 1e-9);
            let r = p / ft;
            let verdict = if r >= 1.0 { format!("FT {r:.2}x FASTER") } else { format!("FT {:.2}x slower", 1.0 / r) };
            println!("   PyTorch {p:8.3} ms  => {verdict}  (rel-diff {rel:.2e} {})",
                if rel < 1e-4 { "MATCH" } else { "MISMATCH!" });
        }
        None => println!("   PyTorch (unavailable)"),
    }
}

fn main() {
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    println!("f32 SDPA forward (RAW sdpa_forward_f32 kernel) [{BH},{SEQ},{D}], {iters} iters MIN:");
    let total = BH * SEQ * D;
    let q = seq_f32(total, 0.0);
    let k = seq_f32(total, 1.0);
    let v = seq_f32(total, 2.0);
    report(&q, &k, &v, "non-causal", false, iters);
    report(&q, &k, &v, "causal", true, iters);
}
