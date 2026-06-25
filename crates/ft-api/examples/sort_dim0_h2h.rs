//! sort along a LEADING dim (dim=0) head-to-head vs PyTorch. sort_tensor parallelizes over OUTER
//! blocks, so dim=0 (outer_size==1) runs serial: it sorts the inner columns sequentially, each an
//! O(n log n)/radix sort. Sorting is compute-bound → parallelizing the independent columns scales.
//! f64 no-grad, dim=0, ascending.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example sort_dim0_h2h

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const R: usize = 4096;
const C: usize = 2048;

fn main() {
    let n = R * C;
    // Pseudo-random, NaN-free (exercises the radix fast path), no exploitable order.
    let data: Vec<f64> = (0..n)
        .map(|i| (((i * 2_654_435_761usize) % 1_000_003) as f64) / 1_000_003.0 - 0.5)
        .collect();
    let mut best = f64::INFINITY;
    let mut chk = 0.0;
    for _ in 0..10 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(data.clone(), vec![R, C], false).unwrap();
        let t = Instant::now();
        let (out, _idx) = s.tensor_sort(x, 0, false).unwrap();
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
            let v = s.tensor_values(out).unwrap();
            // checksum: sum of first row (the per-column minimums) — order-sensitive sanity.
            chk = v[..C].iter().sum();
        }
    }
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
R,C=4096,2048
idx = torch.arange(R*C, dtype=torch.float64)
x = (((idx * 2654435761) % 1000003).double().div(1000003.0).sub(0.5)).reshape(R,C)
for _ in range(3): torch.sort(x, dim=0)
ts=[]; chk=0.0
for _ in range(10):
    t=time.perf_counter(); v,i=torch.sort(x, dim=0); ts.append((time.perf_counter()-t)*1e3); chk=v[0].sum().item()
print("MS", sorted(ts)[0]); print("CHK", chk)
"#;
    let out = Command::new(&python).arg("-c").arg(py).output();
    println!("sort(x, dim=0) [{R},{C}] f64 no-grad, 10 iters MIN:");
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
            let rel = (chk - pc).abs() / (pc.abs() + 1e-9);
            println!("  PyTorch      : {p:8.3} ms   chk {pc:.6e}");
            println!(
                "  CORRECTNESS  : firstrow rel {rel:.2e} ({})",
                if rel < 1e-9 { "MATCH" } else { "MISMATCH!" }
            );
            let r = p / best;
            if r >= 1.0 {
                println!("  => FT {r:.2}x FASTER");
            } else {
                println!("  => FT {:.2}x slower", 1.0 / r);
            }
        }
    }
}
