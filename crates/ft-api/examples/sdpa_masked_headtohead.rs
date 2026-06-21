//! Masked f64 SDPA head-to-head vs PyTorch (BlackThrush). The explicit additive-mask
//! path used to fall through to bmm+softmax+bmm (materialized) like PyTorch's f64 path;
//! the new flash masked kernel (sdpa_forward_masked_f64) folds the mask into the softmax.
//! VERIFIES correctness vs torch (output sums match) AND measures the win. f64, no-grad.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example sdpa_masked_headtohead

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const BH: usize = 16;
const B: usize = 2;
const HQ: usize = 8;
const HKV: usize = 2;
const SEQ: usize = 512;
const D: usize = 64;

fn seq_vals(n: usize, shift: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (((i as f64) * 0.017 + shift).sin()) * 0.2)
        .collect()
}
// mild deterministic additive mask [SEQ,SEQ], no -inf (every row keeps weight -> no NaN)
fn mask_vals() -> Vec<f64> {
    (0..SEQ * SEQ)
        .map(|idx| if idx % 3 == 0 { -0.5 } else { 0.0 })
        .collect()
}

fn ft_masked() -> (f64, f64) {
    let total = BH * SEQ * D;
    let shape = vec![BH, SEQ, D];
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let q = s
        .tensor_variable(seq_vals(total, 0.0), shape.clone(), false)
        .unwrap();
    let k = s
        .tensor_variable(seq_vals(total, 1.0), shape.clone(), false)
        .unwrap();
    let v = s
        .tensor_variable(seq_vals(total, 2.0), shape, false)
        .unwrap();
    let m = s
        .tensor_variable(mask_vals(), vec![SEQ, SEQ], false)
        .unwrap();
    let t = Instant::now();
    let out = s
        .scaled_dot_product_attention(q, k, v, Some(m), 0.0, false)
        .unwrap();
    let vals = s.tensor_values(out).unwrap();
    (vals.iter().sum(), t.elapsed().as_secs_f64() * 1e3)
}

fn ft_masked_tensor_api() -> (f64, f64) {
    let total = BH * SEQ * D;
    let shape = vec![BH, SEQ, D];
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let q = s
        .tensor_variable(seq_vals(total, 0.0), shape.clone(), false)
        .unwrap();
    let k = s
        .tensor_variable(seq_vals(total, 1.0), shape.clone(), false)
        .unwrap();
    let v = s
        .tensor_variable(seq_vals(total, 2.0), shape, false)
        .unwrap();
    let m = s
        .tensor_variable(mask_vals(), vec![SEQ, SEQ], false)
        .unwrap();
    let t = Instant::now();
    let out = s
        .tensor_scaled_dot_product_attention(q, k, v, Some(m), false, None)
        .unwrap();
    let vals = s.tensor_values(out).unwrap();
    (vals.iter().sum(), t.elapsed().as_secs_f64() * 1e3)
}

fn ft_masked_gqa() -> (f64, f64) {
    let q_total = B * HQ * SEQ * D;
    let kv_total = B * HKV * SEQ * D;
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let q = s
        .tensor_variable(seq_vals(q_total, 0.0), vec![B, HQ, SEQ, D], false)
        .unwrap();
    let k = s
        .tensor_variable(seq_vals(kv_total, 1.0), vec![B, HKV, SEQ, D], false)
        .unwrap();
    let v = s
        .tensor_variable(seq_vals(kv_total, 2.0), vec![B, HKV, SEQ, D], false)
        .unwrap();
    let m = s
        .tensor_variable(mask_vals(), vec![SEQ, SEQ], false)
        .unwrap();
    let t = Instant::now();
    let out = s
        .tensor_scaled_dot_product_attention_gqa(q, k, v, Some(m), false, None, true)
        .unwrap();
    let vals = s.tensor_values(out).unwrap();
    (vals.iter().sum(), t.elapsed().as_secs_f64() * 1e3)
}

const PY: &str = r#"
import os, time
import torch
import torch.nn.functional as F
BH, B, HQ, HKV, SEQ, D = 16, 2, 8, 2, 512, 64
total = BH*SEQ*D
def dv(shift):
    return (torch.arange(total, dtype=torch.float64).mul_(0.017).add_(shift).sin_().mul_(0.2).reshape(BH,SEQ,D))
def qv_gqa(shift):
    return (torch.arange(B*HQ*SEQ*D, dtype=torch.float64).mul_(0.017).add_(shift).sin_().mul_(0.2).reshape(B,HQ,SEQ,D))
def kv_gqa(shift):
    return (torch.arange(B*HKV*SEQ*D, dtype=torch.float64).mul_(0.017).add_(shift).sin_().mul_(0.2).reshape(B,HKV,SEQ,D))
m = torch.tensor([(-0.5 if idx%3==0 else 0.0) for idx in range(SEQ*SEQ)], dtype=torch.float64).reshape(SEQ,SEQ)
iters = int(os.environ.get("FT_GAUNTLET_ITERS","15"))
torch.set_num_threads(int(os.environ.get("FT_TORCH_THREADS","8")))
q, k, v = dv(0.0), dv(1.0), dv(2.0)
qg, kg, vg = qv_gqa(0.0), kv_gqa(1.0), kv_gqa(2.0)
def measure(label, f):
    with torch.no_grad():
        for _ in range(3):
            f().sum().item()
        ts=[]; chk=0.0
        for _ in range(iters):
            t0=time.perf_counter()
            o = f()
            chk = o.sum().item()
            ts.append((time.perf_counter()-t0)*1e3)
    ts.sort()
    print(label + "_MS", ts[0])
    print(label + "_CHECKSUM", chk)
with torch.no_grad():
    pass
measure("PRIMARY", lambda: F.scaled_dot_product_attention(q,k,v,attn_mask=m))
measure("TENSOR", lambda: F.scaled_dot_product_attention(q,k,v,attn_mask=m))
measure("GQA", lambda: F.scaled_dot_product_attention(qg,kg,vg,attn_mask=m,enable_gqa=True))
"#;

fn bench_ft(label: &str, iters: usize, f: fn() -> (f64, f64)) -> (f64, f64) {
    for _ in 0..3 {
        let _ = f();
    }
    let mut times = Vec::with_capacity(iters);
    let mut sum = 0.0;
    for _ in 0..iters {
        let (s, t) = f();
        sum = s;
        times.push(t);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("  {label:<18}: {:8.3} ms   checksum {:.6e}", times[0], sum);
    (sum, times[0])
}

fn parsed_field(stdout: &str, field: &str) -> Option<f64> {
    stdout.lines().find_map(|line| {
        line.strip_prefix(field)
            .and_then(|v| v.trim().parse::<f64>().ok())
    })
}

fn print_ratio(label: &str, ft_sum: f64, ft_ms: f64, py_ms: Option<f64>, py_sum: Option<f64>) {
    match (py_ms, py_sum) {
        (Some(p), Some(ps)) => {
            let rel = (ft_sum - ps).abs() / (ps.abs() + 1e-12);
            println!("  PyTorch {label:<10}: {p:8.3} ms   checksum {ps:.6e}");
            println!(
                "  CORRECTNESS {label:<7}: rel-diff {rel:.2e}  ({})",
                if rel < 1e-9 { "MATCH" } else { "MISMATCH!" }
            );
            let r = p / ft_ms;
            if r >= 1.0 {
                println!("  => {label}: FT {r:.2}x FASTER");
            } else {
                println!("  => {label}: FT {:.2}x slower", 1.0 / r);
            }
        }
        _ => println!("  PyTorch {label:<10}: (unavailable)"),
    }
}

fn main() {
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    println!("masked f64 SDPA no-grad, {iters} iters MIN:");
    let (primary_sum, primary_ms) = bench_ft("FrankenTorch primary", iters, ft_masked);
    let (tensor_sum, tensor_ms) = bench_ft("FrankenTorch tensor", iters, ft_masked_tensor_api);
    let (gqa_sum, gqa_ms) = bench_ft("FrankenTorch GQA", iters, ft_masked_gqa);

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let out = Command::new(&python)
        .arg("-c")
        .arg(PY)
        .env("FT_GAUNTLET_ITERS", iters.to_string())
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout);
            print_ratio(
                "primary",
                primary_sum,
                primary_ms,
                parsed_field(&s, "PRIMARY_MS "),
                parsed_field(&s, "PRIMARY_CHECKSUM "),
            );
            print_ratio(
                "tensor",
                tensor_sum,
                tensor_ms,
                parsed_field(&s, "TENSOR_MS "),
                parsed_field(&s, "TENSOR_CHECKSUM "),
            );
            print_ratio(
                "GQA",
                gqa_sum,
                gqa_ms,
                parsed_field(&s, "GQA_MS "),
                parsed_field(&s, "GQA_CHECKSUM "),
            );
        }
        Ok(o) => eprintln!("pytorch failed: {}", String::from_utf8_lossy(&o.stderr)),
        Err(e) => eprintln!("launch failed: {e}"),
    }
}
