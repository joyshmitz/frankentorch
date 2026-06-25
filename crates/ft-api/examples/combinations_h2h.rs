//! torch.combinations(input, r=2) head-to-head vs PyTorch (serial in torch, 183ms@n=3000). FT builds
//! the index mapping via serial recursion then gathers serially through apply_function. f64 no-grad.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example combinations_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const NV: usize = 3000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<f64> = (0..NV).map(|i| i as f64).collect();
    let mut best = f64::INFINITY;
    let mut chk = 0.0;
    for _ in 0..5 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(data.clone(), vec![NV], false)?;
        let t = Instant::now();
        let out = s.tensor_combinations(x, 2, false)?;
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
            chk = s.tensor_values(out)?.iter().sum();
        }
    }
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = format!(
        r#"
import time, torch
torch.set_num_threads(8)
x = torch.arange({NV}, dtype=torch.float64)
for _ in range(2): torch.combinations(x, r=2)
ts=[]; chk=0.0
for _ in range(5):
    s=time.perf_counter(); o=torch.combinations(x, r=2); ts.append((time.perf_counter()-s)*1e3); chk=o.sum().item()
print("MS", sorted(ts)[0]); print("CHK", chk)
"#
    );
    let mut child = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| std::io::Error::other("python stdin unavailable"))?
        .write_all(py.as_bytes())?;
    let out = child.wait_with_output();
    println!("combinations(arange({NV}), r=2) f64 no-grad, 5 iters MIN:");
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
            let ok = (chk - pc).abs() / (pc.abs() + 1e-9) < 1e-12;
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
    Ok(())
}
