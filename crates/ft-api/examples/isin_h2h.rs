//! torch.isin(elements, test_elements) head-to-head vs PyTorch. torch.isin is SERIAL (measured FLAT
//! @8 vs @32). FT's tensor_isin is O(n*m) AND serial (per-element linear scan of test_elements).
//! For a large test set this is catastrophic. f64 no-grad.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example isin_h2h

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const N: usize = 4_000_000;
const M: usize = 5000;

fn main() {
    let elems: Vec<f64> = (0..N).map(|i| (i % 10007) as f64).collect();
    let test: Vec<f64> = (0..M).map(|i| i as f64).collect();
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let mut best = f64::INFINITY;
    let mut chk = 0.0;
    for _ in 0..iters {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let e = s.tensor_variable(elems.clone(), vec![N], false).unwrap();
        let t = s.tensor_variable(test.clone(), vec![M], false).unwrap();
        let ti = Instant::now();
        let out = s.tensor_isin(e, t).unwrap();
        let el = ti.elapsed().as_secs_f64() * 1e3;
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
N,M={N},{M}
e = (torch.arange(N) % 10007).double()
t = torch.arange(M).double()
for _ in range(2): torch.isin(e, t)
ts=[]; chk=0.0
for _ in range(5):
    s=time.perf_counter(); o=torch.isin(e, t); ts.append((time.perf_counter()-s)*1e3); chk=o.double().sum().item()
print("MS", sorted(ts)[0]); print("CHK", chk)
"#
    );
    let out = Command::new(&python).arg("-c").arg(&py).output();
    println!("isin(elements[{N}], test[{M}]) f64 no-grad, {iters} iters MIN:");
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
            let ok = (chk - pc).abs() < 0.5;
            let r = p / best;
            let verdict = if r >= 1.0 {
                format!("FT {r:.2}x FASTER")
            } else {
                format!("FT {:.2}x slower", 1.0 / r)
            };
            println!(
                "  PyTorch      : {p:9.2} ms   chk {pc:.6e}  => {verdict}  [{}]",
                if ok { "MATCH" } else { "MISMATCH!" }
            );
        }
    }
}
