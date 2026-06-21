use std::error::Error;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

fn fill_input(batch: usize, m: usize, n: usize) -> Vec<f32> {
    let mut data = vec![0.0_f32; batch * m * n];
    for (idx, slot) in data.iter_mut().enumerate() {
        let plane = idx / (m * n);
        let rem = idx % (m * n);
        let r = rem / n;
        let c = rem % n;
        let low_rank = ((r + 1) * (c + 3) + (plane % 17)) as f32 * 0.0007;
        let noise = ((((plane + 5) * (r + 7) * (c + 11)) % 29) as f32 - 14.0) * 0.002;
        *slot = low_rank
            + noise
            + if r == c {
                2.0 + (plane % 5) as f32 * 0.01
            } else {
                0.0
            };
    }
    data
}

fn run_ft(batch: usize, m: usize, n: usize) -> Result<(f64, f64), Box<dyn Error>> {
    let input = fill_input(batch, m, n);
    let mut best = f64::INFINITY;
    let mut checksum = 0.0_f64;
    for _ in 0..5 {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable_f32(input.clone(), vec![batch, m, n], false)
            .map_err(boxed)?;
        let start = Instant::now();
        let s = session.tensor_linalg_svdvals(x).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
            checksum = session
                .tensor_values_f32(s)
                .map_err(boxed)?
                .iter()
                .map(|&v| f64::from(v))
                .sum();
        }
    }
    Ok((best, checksum))
}

fn run_pytorch(batch: usize, m: usize, n: usize) -> Option<(f64, f64)> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let script = format!(
        r#"
import time, torch
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
B, M, N = {batch}, {m}, {n}
idx = torch.arange(B*M*N, dtype=torch.int64)
p = idx // (M*N)
rem = idx % (M*N)
r = rem // N
c = rem % N
low_rank = ((r + 1) * (c + 3) + (p % 17)).to(torch.float32) * 0.0007
noise = ((((p + 5) * (r + 7) * (c + 11)) % 29).to(torch.float32) - 14.0) * 0.002
diag = torch.where(r == c, 2.0 + (p % 5).to(torch.float32) * 0.01, torch.zeros_like(low_rank))
A = (low_rank + noise + diag).reshape(B, M, N)
for _ in range(2):
    torch.linalg.svdvals(A)
samples = []
for _ in range(5):
    t = time.perf_counter()
    torch.linalg.svdvals(A)
    samples.append((time.perf_counter() - t) * 1e3)
s = torch.linalg.svdvals(A)
print("MS", min(samples))
print("SUM", s.double().sum().item())
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
    for (batch, m, n) in [
        (100_000usize, 8usize, 4usize),
        (20_000usize, 16usize, 8usize),
        (8_000usize, 32usize, 16usize),
    ] {
        let (ft_ms, ft_sum) = run_ft(batch, m, n)?;
        print!("B={batch} m={m} n={n}: FT {ft_ms:.3} ms sum {ft_sum:.9e}");
        if let Some((torch_ms, torch_sum)) = run_pytorch(batch, m, n) {
            let rel = (ft_sum - torch_sum).abs() / (torch_sum.abs() + 1e-12);
            let ratio = torch_ms / ft_ms;
            let verdict = if ratio >= 1.0 {
                format!("FT {ratio:.2}x FASTER")
            } else {
                format!("FT {:.2}x slower", 1.0 / ratio)
            };
            println!(" | PyTorch {torch_ms:.3} ms sum {torch_sum:.9e} rel {rel:.3e} | {verdict}");
        } else {
            println!(" | PyTorch unavailable");
        }
    }
    Ok(())
}
