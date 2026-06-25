//! torch.nn.functional.fold (col2im) head-to-head vs PyTorch. torch's fold is SERIAL (~2.2s @ this
//! shape, FLAT across thread counts). FT's tensor_fold forward was a serial nested accumulation;
//! parallelized over the (n,c) lanes (disjoint output blocks). f64 no-grad.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example fold_h2h

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const N: usize = 64;
const C: usize = 64;
const KH: usize = 3;
const KW: usize = 3;
const OH: usize = 56;
const OW: usize = 56;

fn main() {
    let l = (OH - KH + 1) * (OW - KW + 1); // 54*54 = 2916
    let blocks = C * KH * KW; // 576
    let data: Vec<f64> = (0..N * blocks * l)
        .map(|i| ((i as f64) * 1e-6).sin())
        .collect();
    let mut best = f64::INFINITY;
    let mut chk = 0.0;
    for _ in 0..5 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(data.clone(), vec![N, blocks, l], false)
            .unwrap();
        let t = Instant::now();
        let out = s
            .tensor_fold(x, (OH, OW), (KH, KW), (1, 1), (0, 0), (1, 1))
            .unwrap();
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
            chk = s.tensor_values(out).unwrap().iter().sum();
        }
    }
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = format!(
        r#"
import time, torch
torch.set_num_threads(8)
N,C,KH,KW,OH,OW={N},{C},{KH},{KW},{OH},{OW}
L=(OH-KH+1)*(OW-KW+1)
x=(torch.arange(N*C*KH*KW*L, dtype=torch.float64).mul_(1e-6).sin_()).reshape(N,C*KH*KW,L)
fn=lambda: torch.nn.functional.fold(x,(OH,OW),(KH,KW))
for _ in range(2): fn()
ts=[]; chk=0.0
for _ in range(5):
    t=time.perf_counter(); o=fn(); ts.append((time.perf_counter()-t)*1e3); chk=o.sum().item()
print("MS", sorted(ts)[0]); print("CHK", chk)
"#
    );
    let out = Command::new(&python).arg("-c").arg(&py).output();
    println!("fold([{N},{blocks},{l}] -> [{N},{C},{OH},{OW}]) f64 no-grad, 5 iters MIN:");
    println!("  FrankenTorch : {best:9.2} ms   chk {chk:.6e}");
    if let Ok(o) = out
        && o.status.success()
    {
        let s = String::from_utf8_lossy(&o.stdout);
        let g = |p: &str| {
            s.lines()
                .find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()))
        };
        if let (Some(p), Some(pc)) = (g("MS "), g("CHK ")) {
            let rel = (chk - pc).abs() / (pc.abs() + 1e-9);
            let r = p / best;
            let verdict = if r >= 1.0 {
                format!("FT {r:.2}x FASTER")
            } else {
                format!("FT {:.2}x slower", 1.0 / r)
            };
            println!(
                "  PyTorch      : {p:9.2} ms   chk {pc:.6e}  => {verdict}  [{}]",
                if rel < 1e-9 { "MATCH" } else { "MISMATCH!" }
            );
        }
    }
}
