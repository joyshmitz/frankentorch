//! `tensor_normalize` no-grad f64 head-to-head vs PyTorch.
//!
//! Run:
//!   PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example normalize_h2h

use std::error::Error;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

fn data(rows: usize, cols: usize) -> Vec<f64> {
    (0..rows * cols)
        .map(|i| ((i as f64 * 0.013).sin() * 0.5) + 0.25)
        .collect()
}

fn run_ft(rows: usize, cols: usize, reps: usize) -> Result<(f64, f64), Box<dyn Error>> {
    let mut best = f64::INFINITY;
    let mut checksum = 0.0;
    for _ in 0..reps {
        let values = data(rows, cols);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(values, vec![rows, cols], false)
            .map_err(boxed)?;
        let start = Instant::now();
        let out = s.tensor_normalize(x, 2.0, 1, 1e-12).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        let vals = s.tensor_values(out).map_err(boxed)?;
        if elapsed_ms < best {
            best = elapsed_ms;
            checksum = vals.iter().sum();
        }
    }
    Ok((best, checksum))
}

fn run_pytorch(rows: usize, cols: usize, reps: usize) -> Option<(f64, f64)> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let script = format!(
        r#"
import time, torch
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
R,C,REPS={rows},{cols},{reps}
i=torch.arange(R*C,dtype=torch.float64)
x=(i.mul(0.013).sin().mul(0.5).add(0.25)).reshape(R,C)
def step():
    return torch.nn.functional.normalize(x, p=2.0, dim=1, eps=1e-12)
with torch.no_grad():
    for _ in range(2):
        step().sum().item()
    times=[]
    total=0.0
    for _ in range(REPS):
        t=time.perf_counter()
        y=step()
        times.append((time.perf_counter()-t)*1e3)
        total=float(y.sum().item())
print("MS", min(times))
print("SUM", total)
"#
    );
    let mut child = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.as_mut()?.write_all(script.as_bytes()).ok()?;
    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let get = |prefix: &str| {
        stdout
            .lines()
            .find_map(|line| line.strip_prefix(prefix))
            .and_then(|value| value.trim().parse::<f64>().ok())
    };
    Some((get("MS ")?, get("SUM ")?))
}

fn main() -> Result<(), Box<dyn Error>> {
    let reps = std::env::var("NORMALIZE_H2H_REPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(5);
    let (rows, cols) = (4_000usize, 4_000usize);
    let (ft_ms, ft_sum) = run_ft(rows, cols, reps)?;
    print!("normalize f64 dim=1 [{rows},{cols}]: FT {ft_ms:.3} ms sum {ft_sum:.6e}");
    if let Some((torch_ms, torch_sum)) = run_pytorch(rows, cols, reps) {
        let rel = (ft_sum - torch_sum).abs() / (torch_sum.abs() + 1e-12);
        let ratio = torch_ms / ft_ms;
        if ratio >= 1.0 {
            println!(
                " | PyTorch {torch_ms:.3} ms sum {torch_sum:.6e} rel {rel:.3e} | FT {ratio:.2}x FASTER"
            );
        } else {
            println!(
                " | PyTorch {torch_ms:.3} ms sum {torch_sum:.6e} rel {rel:.3e} | FT {:.2}x slower",
                1.0 / ratio
            );
        }
    } else {
        println!(" | PyTorch unavailable");
    }
    Ok(())
}
