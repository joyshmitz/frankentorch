//! torch.unique(sorted=True) head-to-head vs PyTorch. torch.unique is SERIAL (O(n log n) sort-based,
//! measured FLAT across 8/32 threads). FT's tensor_unique dedups in O(n) via a hash map, then sorts
//! only the (typically far smaller) set of unique values. For large inputs FT's algorithm wins.
//! Two regimes: few-unique (503 distinct) and all-distinct. f64 no-grad.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example unique_h2h

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const N: usize = 8192 * 4096; // ~33.5M

fn bench(label: &str, data: &[f64], py_gen: &str) {
    let mut best = f64::INFINITY;
    let mut ulen = 0.0;
    let mut usum = 0.0;
    for _ in 0..6 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(data.to_vec(), vec![data.len()], false)
            .unwrap();
        let t = Instant::now();
        let (u, _inv, _cnt) = s.tensor_unique(x, true, false, false).unwrap();
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
            let uv = s.tensor_values(u).unwrap();
            ulen = uv.len() as f64;
            usum = uv.iter().sum();
        }
    }
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = format!(
        r#"
import time, torch
torch.set_num_threads(8)
{py_gen}
for _ in range(2): torch.unique(x, sorted=True)
ts=[]; ul=0; us=0.0
for _ in range(6):
    t=time.perf_counter(); u=torch.unique(x, sorted=True); ts.append((time.perf_counter()-t)*1e3); ul=u.numel(); us=u.sum().item()
print("MS", sorted(ts)[0]); print("UL", ul); print("US", us)
"#
    );
    let out = Command::new(&python).arg("-c").arg(&py).output();
    print!("{label:14}: FT {best:9.2} ms  (uniq {ulen:.0})");
    if let Ok(o) = out
        && o.status.success()
    {
        let s = String::from_utf8_lossy(&o.stdout);
        let g = |p: &str| {
            s.lines()
                .find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()))
        };
        if let (Some(p), Some(pl), Some(ps)) = (g("MS "), g("UL "), g("US ")) {
            let ok = (ulen - pl).abs() < 0.5 && (usum - ps).abs() / (ps.abs() + 1e-9) < 1e-9;
            let r = p / best;
            let verdict = if r >= 1.0 {
                format!("FT {r:.2}x FASTER")
            } else {
                format!("FT {:.2}x slower", 1.0 / r)
            };
            println!(
                "  | torch {p:9.2} ms (uniq {pl:.0}) => {verdict}  [{}]",
                if ok { "MATCH" } else { "MISMATCH!" }
            );
        } else {
            println!("  (torch parse fail)");
        }
    } else {
        println!("  (torch fail)");
    }
}

fn main() {
    // Regime A: few unique (503 distinct).
    let few: Vec<f64> = (0..N).map(|i| (i % 503) as f64).collect();
    bench(
        "few-unique-503",
        &few,
        "N=8192*4096\nx=(torch.arange(N)%503).double()",
    );
    // Regime B: all distinct.
    let distinct: Vec<f64> = (0..N).map(|i| i as f64).collect();
    bench(
        "all-distinct",
        &distinct,
        "N=8192*4096\nx=torch.arange(N).double()",
    );
}
