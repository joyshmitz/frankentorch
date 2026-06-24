//! cumprod strided-dim (dim=0) transpose trick. cumprod along a LEADING dim of a 2-D tensor has
//! outer_size==1, so FT's outer-block rayon fan-out leaves the pool idle and runs a single serial
//! lane block over an inner_size-wide accumulator (bandwidth-bound, one core). The fix (same as
//! cumsum): scan the inner_size INDEPENDENT lanes across the pool by transposing the scan dim to
//! innermost, scanning, transposing back. Per-lane multiplicative order is preserved => bit-exact.
//! Measures FT-direct vs FT-transpose-trick(explicit ops) vs PyTorch. f64, no-grad.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example cumprod_transpose_trick

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const ROWS: usize = 2048;
const COLS: usize = 2048;

fn data() -> Vec<f64> {
    // Values near 1.0 so a 2048-long product stays finite (and matches torch bit-for-bit).
    (0..ROWS * COLS)
        .map(|i| 1.0 + ((i % 1000) as f64) * 1e-6)
        .collect()
}

fn ft_direct(d: &[f64], iters: usize) -> (f64, f64) {
    let mut best = f64::INFINITY;
    let mut chk = 0.0;
    for _ in 0..iters + 3 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(d.to_vec(), vec![ROWS, COLS], false)
            .unwrap();
        let t = Instant::now();
        let out = s.tensor_cumprod(x, 0).unwrap();
        let el = t.elapsed().as_secs_f64() * 1e3;
        chk = s.tensor_values(out).unwrap().iter().sum();
        if el < best {
            best = el;
        }
    }
    (chk, best)
}

fn ft_trick(d: &[f64], iters: usize) -> (f64, f64) {
    let mut best = f64::INFINITY;
    let mut chk = 0.0;
    for _ in 0..iters + 3 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(d.to_vec(), vec![ROWS, COLS], false)
            .unwrap();
        let t = Instant::now();
        let xt = s.tensor_transpose(x, 0, 1).unwrap();
        let xtc = s.tensor_contiguous(xt).unwrap();
        let cs = s.tensor_cumprod(xtc, 1).unwrap();
        let back = s.tensor_transpose(cs, 0, 1).unwrap();
        let out = s.tensor_contiguous(back).unwrap();
        let el = t.elapsed().as_secs_f64() * 1e3;
        chk = s.tensor_values(out).unwrap().iter().sum();
        if el < best {
            best = el;
        }
    }
    (chk, best)
}

const PY: &str = r#"
import time, torch
torch.set_num_threads(8)
ROWS, COLS = 2048, 2048
idx = torch.arange(ROWS*COLS, dtype=torch.float64)
x = (1.0 + (idx % 1000).double().mul(1e-6)).reshape(ROWS, COLS)
for _ in range(3): torch.cumprod(x, dim=0)
ts=[]; chk=0.0
for _ in range(15):
    t=time.perf_counter(); o=torch.cumprod(x, dim=0); ts.append((time.perf_counter()-t)*1e3); chk=o.sum().item()
print("MS", sorted(ts)[0]); print("CHK", chk)
"#;

fn main() {
    let d = data();
    let (sd, direct) = ft_direct(&d, 15);
    let (st, trick) = ft_trick(&d, 15);
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let o = Command::new(&python).arg("-c").arg(PY).output();
    let (py_ms, py_sum) = match o {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout);
            let g = |p: &str| {
                s.lines()
                    .find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()))
            };
            (g("MS "), g("CHK "))
        }
        _ => (None, None),
    };
    println!("cumprod [2048,2048] dim=0 (leading) f64, 15 iters MIN:");
    println!("  FT direct        : {direct:8.3} ms");
    println!(
        "  FT transpose-trick: {trick:8.3} ms   (bit-exact vs direct: {})",
        if (sd - st).abs() / (sd.abs() + 1e-12) < 1e-12 {
            "MATCH"
        } else {
            "MISMATCH!"
        }
    );
    if let (Some(p), Some(ps)) = (py_ms, py_sum) {
        let corr = (sd - ps).abs() / (ps.abs() + 1e-12);
        println!(
            "  PyTorch dim=0    : {p:8.3} ms   (FT-direct correct vs torch: {} {corr:.1e})",
            if corr < 1e-9 { "MATCH" } else { "MISMATCH!" }
        );
        let r = p / direct;
        if r >= 1.0 {
            println!("  => FT direct {r:.2}x FASTER than PyTorch");
        } else {
            println!("  => FT direct {:.2}x slower than PyTorch", 1.0 / r);
        }
    }
}
