//! torch.mode(x, dim=-1) head-to-head vs PyTorch. FT's tensor_mode loops its outer slices SERIALLY
//! (no rayon), each an O(M log M) sort + run-length count. The slices are independent → parallelize.
//! f64 no-grad, mode along the last dim. Values drawn from a small range to create real ties.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example mode_h2h

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const R: usize = 4096;
const C: usize = 4096;

fn main() {
    // Small value range -> many ties -> mode is meaningful and the count loop matters.
    let data: Vec<f64> = (0..R * C)
        .map(|i| (((i * 1_103_515_245usize + 12345) % 97) as f64))
        .collect();
    let mut best = f64::INFINITY;
    let mut chk = 0.0;
    for _ in 0..8 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(data.clone(), vec![R, C], false).unwrap();
        let t = Instant::now();
        let (vals, _idx) = s.tensor_mode(x).unwrap();
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
            chk = s.tensor_values(vals).unwrap().iter().sum();
        }
    }
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
R,C=4096,4096
idx = torch.arange(R*C, dtype=torch.int64)
x = (((idx * 1103515245 + 12345) % 97)).double().reshape(R,C)
for _ in range(2): torch.mode(x, dim=-1)
ts=[]; chk=0.0
for _ in range(8):
    t=time.perf_counter(); v,i=torch.mode(x, dim=-1); ts.append((time.perf_counter()-t)*1e3); chk=v.sum().item()
print("MS", sorted(ts)[0]); print("CHK", chk)
"#;
    let out = Command::new(&python).arg("-c").arg(py).output();
    println!("mode(x, dim=-1) [{R},{C}] f64 no-grad, 8 iters MIN:");
    println!("  FrankenTorch : {best:8.3} ms   chk {chk:.6e}");
    if let Ok(o) = out
        && o.status.success()
    {
        let s = String::from_utf8_lossy(&o.stdout);
        let g = |p: &str| {
            s.lines()
                .find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()))
        };
        if let (Some(p), Some(pc)) = (g("MS "), g("CHK ")) {
            // mode VALUE can differ from torch on ties (tie-break rule), so report rel but don't gate.
            let rel = (chk - pc).abs() / (pc.abs() + 1e-9);
            println!("  PyTorch      : {p:8.3} ms   chk {pc:.6e}  (value-sum rel {rel:.1e})");
            let r = p / best;
            if r >= 1.0 {
                println!("  => FT {r:.2}x FASTER");
            } else {
                println!("  => FT {:.2}x SLOWER", 1.0 / r);
            }
        }
    }
}
