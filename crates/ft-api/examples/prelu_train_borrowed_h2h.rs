//! PReLU train-step head-to-head vs PyTorch.
//!
//! Run:
//!   PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example prelu_train_borrowed_h2h

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const BATCH: usize = 32;
const CHANNELS: usize = 512;
const WIDTH: usize = 256;
const TOTAL: usize = BATCH * CHANNELS * WIDTH;

fn input_values() -> Vec<f64> {
    (0..TOTAL)
        .map(|i| {
            let x = (i as f64).mul_add(0.013, 0.37).sin();
            x.mul_add(0.85, -0.08)
        })
        .collect()
}

fn weight_values() -> Vec<f64> {
    (0..CHANNELS)
        .map(|i| (i as f64).mul_add(0.0007, 0.04))
        .collect()
}

fn ft_prelu_step(input: &[f64], weight: &[f64]) -> f64 {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s
        .tensor_variable(input.to_vec(), vec![BATCH, CHANNELS, WIDTH], true)
        .unwrap();
    let w = s
        .tensor_variable(weight.to_vec(), vec![CHANNELS], true)
        .unwrap();
    let out = s.tensor_prelu(x, w).unwrap();
    let loss = s.tensor_sum(out).unwrap();
    let report = s.tensor_backward(loss).unwrap();
    let gx: f64 = report.gradient(x).unwrap().iter().map(|v| v.abs()).sum();
    let gw: f64 = report.gradient(w).unwrap().iter().map(|v| v.abs()).sum();
    gx + gw
}

const PY: &str = r#"
import os, time
import torch
import torch.nn.functional as F

BATCH, CHANNELS, WIDTH = 32, 512, 256
TOTAL = BATCH * CHANNELS * WIDTH
ITERS = int(os.environ.get("FT_GAUNTLET_ITERS", "20"))
torch.set_num_threads(int(os.environ.get("FT_TORCH_THREADS", "8")))

base_x = torch.arange(TOTAL, dtype=torch.float64).mul_(0.013).add_(0.37).sin_().mul_(0.85).add_(-0.08).reshape(BATCH, CHANNELS, WIDTH)
base_w = torch.arange(CHANNELS, dtype=torch.float64).mul_(0.0007).add_(0.04)

def step():
    x = base_x.detach().clone().requires_grad_(True)
    w = base_w.detach().clone().requires_grad_(True)
    F.prelu(x, w).sum().backward()
    return x.grad.abs().sum().item() + w.grad.abs().sum().item()

for _ in range(3):
    step()

times = []
checksum = 0.0
for _ in range(ITERS):
    t0 = time.perf_counter()
    checksum = step()
    times.append((time.perf_counter() - t0) * 1e3)
times.sort()
print("ELAPSED_MS", times[len(times)//2])
print("CHECKSUM", checksum)
"#;

fn bench_ft(iters: usize) -> (f64, f64) {
    let input = input_values();
    let weight = weight_values();
    for _ in 0..3 {
        std::hint::black_box(ft_prelu_step(&input, &weight));
    }
    let mut times = Vec::with_capacity(iters);
    let mut checksum = 0.0;
    for _ in 0..iters {
        let t0 = Instant::now();
        checksum = ft_prelu_step(&input, &weight);
        times.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[times.len() / 2], checksum)
}

fn py_result(iters: usize) -> Option<(f64, f64)> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let out = Command::new(&python)
        .arg("-c")
        .arg(PY)
        .env("FT_GAUNTLET_ITERS", iters.to_string())
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!("pytorch failed: {}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    let mut elapsed = None;
    let mut checksum = None;
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        if let Some(v) = line.strip_prefix("ELAPSED_MS ") {
            elapsed = v.trim().parse::<f64>().ok();
        } else if let Some(v) = line.strip_prefix("CHECKSUM ") {
            checksum = v.trim().parse::<f64>().ok();
        }
    }
    elapsed.zip(checksum)
}

fn main() {
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let (ft, ft_checksum) = bench_ft(iters);
    println!("PReLU f64 train step [{BATCH},{CHANNELS},{WIDTH}], {iters} iters median:");
    print!("  FrankenTorch FT {ft:8.3} ms  checksum {ft_checksum:.6}");
    if let Some((py, py_checksum)) = py_result(iters) {
        let rel = (ft_checksum - py_checksum).abs() / py_checksum.abs().max(1.0);
        let ratio = py / ft;
        if ratio >= 1.0 {
            println!(
                "   PyTorch {py:8.3} ms  checksum {py_checksum:.6}  rel {rel:.3e}  => FT {ratio:.2}x FASTER"
            );
        } else {
            println!(
                "   PyTorch {py:8.3} ms  checksum {py_checksum:.6}  rel {rel:.3e}  => FT {:.2}x slower",
                1.0 / ratio
            );
        }
    } else {
        println!("   PyTorch (unavailable)");
    }
}
