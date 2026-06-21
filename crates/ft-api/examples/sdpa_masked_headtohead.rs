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
const SEQ: usize = 512;
const D: usize = 64;

fn seq_vals(n: usize, shift: f64) -> Vec<f64> {
    (0..n).map(|i| (((i as f64) * 0.017 + shift).sin()) * 0.2).collect()
}
// mild deterministic additive mask [SEQ,SEQ], no -inf (every row keeps weight -> no NaN)
fn mask_vals() -> Vec<f64> {
    (0..SEQ * SEQ).map(|idx| if idx % 3 == 0 { -0.5 } else { 0.0 }).collect()
}

fn ft_masked() -> (f64, f64) {
    let total = BH * SEQ * D;
    let shape = vec![BH, SEQ, D];
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let q = s.tensor_variable(seq_vals(total, 0.0), shape.clone(), false).unwrap();
    let k = s.tensor_variable(seq_vals(total, 1.0), shape.clone(), false).unwrap();
    let v = s.tensor_variable(seq_vals(total, 2.0), shape, false).unwrap();
    let m = s.tensor_variable(mask_vals(), vec![SEQ, SEQ], false).unwrap();
    let t = Instant::now();
    let out = s.scaled_dot_product_attention(q, k, v, Some(m), 0.0, false).unwrap();
    let vals = s.tensor_values(out).unwrap();
    (vals.iter().sum(), t.elapsed().as_secs_f64() * 1e3)
}

const PY: &str = r#"
import os, time
import torch
import torch.nn.functional as F
BH, SEQ, D = 16, 512, 64
total = BH*SEQ*D
def dv(shift):
    return (torch.arange(total, dtype=torch.float64).mul_(0.017).add_(shift).sin_().mul_(0.2).reshape(BH,SEQ,D))
m = torch.tensor([(-0.5 if idx%3==0 else 0.0) for idx in range(SEQ*SEQ)], dtype=torch.float64).reshape(SEQ,SEQ)
iters = int(os.environ.get("FT_GAUNTLET_ITERS","15"))
torch.set_num_threads(int(os.environ.get("FT_TORCH_THREADS","8")))
q, k, v = dv(0.0), dv(1.0), dv(2.0)
with torch.no_grad():
    for _ in range(3):
        F.scaled_dot_product_attention(q,k,v,attn_mask=m).sum().item()
    ts=[]; chk=0.0
    for _ in range(iters):
        t0=time.perf_counter()
        o = F.scaled_dot_product_attention(q,k,v,attn_mask=m)
        chk = o.sum().item()
        ts.append((time.perf_counter()-t0)*1e3)
ts.sort()
print("ELAPSED_MS", ts[0]); print("CHECKSUM", chk)
"#;

fn main() {
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(15);
    for _ in 0..3 {
        let _ = ft_masked();
    }
    let mut times = Vec::with_capacity(iters);
    let mut ft_sum = 0.0;
    for _ in 0..iters {
        let (s, t) = ft_masked();
        ft_sum = s;
        times.push(t);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ft = times[0];

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let out = Command::new(&python).arg("-c").arg(PY)
        .env("FT_GAUNTLET_ITERS", iters.to_string()).output();
    let (py_ms, py_sum) = match out {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout);
            let g = |p: &str| s.lines().find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()));
            (g("ELAPSED_MS "), g("CHECKSUM "))
        }
        Ok(o) => { eprintln!("pytorch failed: {}", String::from_utf8_lossy(&o.stderr)); (None, None) }
        Err(e) => { eprintln!("launch failed: {e}"); (None, None) }
    };

    println!("masked f64 SDPA [{BH},{SEQ},{D}] no-grad, {iters} iters MIN:");
    println!("  FrankenTorch : {ft:8.3} ms   checksum {ft_sum:.6e}");
    match (py_ms, py_sum) {
        (Some(p), Some(ps)) => {
            let rel = (ft_sum - ps).abs() / (ps.abs() + 1e-12);
            println!("  PyTorch      : {p:8.3} ms   checksum {ps:.6e}");
            println!("  CORRECTNESS  : rel-diff {rel:.2e}  ({})", if rel < 1e-9 { "MATCH (flash masked == torch)" } else { "MISMATCH!" });
            let r = p / ft;
            if r >= 1.0 { println!("  => FT {r:.2}x FASTER (masked SDPA)"); }
            else { println!("  => FT {:.2}x slower", 1.0 / r); }
        }
        _ => println!("  PyTorch      : (unavailable)"),
    }
}
