//! End-to-end `tensor_topk` dim-0 head-to-head vs PyTorch.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example topk_dim0_h2h

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
    let n = rows * cols;
    (0..n)
        .map(|i| {
            let x =
                ((i as u64).wrapping_mul(1_103_515_245).wrapping_add(12_345) % 1_000_003) as f64;
            x * 0.001 - 500.0
        })
        .collect()
}

fn run_ft(
    rows: usize,
    cols: usize,
    k: usize,
    reps: usize,
) -> Result<(f64, f64, f64), Box<dyn Error>> {
    let values = data(rows, cols);
    let mut best = f64::INFINITY;
    let mut value_sum = 0.0;
    let mut index_sum = 0.0;
    for _ in 0..reps {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(values.clone(), vec![rows, cols], false)
            .map_err(boxed)?;
        let start = Instant::now();
        let (top, indices) = s.tensor_topk(x, k, 0, true, true).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
            value_sum = s.tensor_values(top).map_err(boxed)?.iter().sum();
            index_sum = indices.iter().map(|&i| i as f64).sum();
        }
    }
    Ok((best, value_sum, index_sum))
}

fn run_pytorch(rows: usize, cols: usize, k: usize, reps: usize) -> Option<(f64, f64, f64)> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let script = format!(
        r#"
import time, torch
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
R,C,K,REPS={rows},{cols},{k},{reps}
n=R*C
i=torch.arange(n,dtype=torch.int64)
x=((i*1103515245 + 12345) % 1000003).to(torch.float64).mul_(0.001).sub_(500.0).reshape(R,C)
def step():
    return torch.topk(x, K, dim=0, largest=True, sorted=True)
for _ in range(2):
    step()
times=[]
for _ in range(REPS):
    t=time.perf_counter()
    values, indices = step()
    times.append((time.perf_counter()-t)*1e3)
print("MS", min(times))
print("VSUM", float(values.sum().item()))
print("ISUM", float(indices.to(torch.float64).sum().item()))
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
    Some((get("MS ")?, get("VSUM ")?, get("ISUM ")?))
}

fn main() -> Result<(), Box<dyn Error>> {
    let reps = std::env::var("TOPK_DIM0_H2H_REPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3);
    let (rows, cols, k) = (262_144usize, 64usize, 8usize);
    let (ft_ms, ft_vsum, ft_isum) = run_ft(rows, cols, k, reps)?;
    print!(
        "tensor_topk dim=0 [{rows},{cols}] k={k}: FT {ft_ms:.3} ms vsum {ft_vsum:.6e} isum {ft_isum:.6e}"
    );
    if let Some((torch_ms, torch_vsum, torch_isum)) = run_pytorch(rows, cols, k, reps) {
        let vrel = (ft_vsum - torch_vsum).abs() / (torch_vsum.abs() + 1e-12);
        let irel = (ft_isum - torch_isum).abs() / (torch_isum.abs() + 1e-12);
        let ratio = torch_ms / ft_ms;
        if ratio >= 1.0 {
            println!(
                " | PyTorch {torch_ms:.3} ms vsum {torch_vsum:.6e} isum {torch_isum:.6e} rel {vrel:.3e}/{irel:.3e} | FT {ratio:.2}x FASTER"
            );
        } else {
            println!(
                " | PyTorch {torch_ms:.3} ms vsum {torch_vsum:.6e} isum {torch_isum:.6e} rel {vrel:.3e}/{irel:.3e} | FT {:.2}x slower",
                1.0 / ratio
            );
        }
    } else {
        println!(" | PyTorch unavailable");
    }
    Ok(())
}
