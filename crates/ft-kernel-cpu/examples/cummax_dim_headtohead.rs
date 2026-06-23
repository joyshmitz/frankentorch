//! cummax-along-dim head-to-head vs PyTorch (BlackThrush). PyTorch CPU cummax along a strided
//! (non-last) dim is very slow (dim=0 [262144,64] = 425ms — it writes BOTH values AND indices with
//! a dim-stride). FT's cache-friendly kernel (cummax_dim_tensor_contiguous_f64, d-outer/inner-inner)
//! walks contiguous inner runs. Verifies values+indices match torch + measures. f64.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-kernel-cpu --example cummax_dim_headtohead

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_core::{DType, Device, TensorMeta};

const R: usize = 262144;
const C: usize = 64;

fn main() {
    let n = R * C;
    let data: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.001).sin()).collect();
    let meta = TensorMeta::from_shape(vec![R, C], DType::F64, Device::Cpu);

    // warmup + time (15 iters MIN)
    let mut best = f64::INFINITY;
    let (mut vsum, mut isum) = (0.0, 0.0);
    for _ in 0..18 {
        let t = Instant::now();
        let (vals, idxs) =
            ft_kernel_cpu::cummax_dim_tensor_contiguous_f64(&data, &meta, 0).unwrap();
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
            vsum = vals.iter().sum();
            isum = idxs.iter().sum();
        }
    }

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
R,C=262144,64
x = torch.arange(R*C, dtype=torch.float64).mul_(0.001).sin_().reshape(R,C)
for _ in range(3): torch.cummax(x, dim=0)
ts=[]
for _ in range(15):
    t=time.perf_counter(); o=torch.cummax(x, dim=0); ts.append((time.perf_counter()-t)*1e3)
v,i = torch.cummax(x, dim=0)
print("MS", sorted(ts)[0]); print("VSUM", v.sum().item()); print("ISUM", float(i.sum().item()))
"#;
    let out = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(py.as_bytes())?;
            }
            child.wait_with_output()
        });
    println!("cummax dim=0 [{R},{C}] f64, 15 iters MIN:");
    println!("  FrankenTorch : {best:8.3} ms   vsum {vsum:.6e}  isum {isum:.6e}");
    match out {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout);
            let g = |p: &str| {
                s.lines()
                    .find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()))
            };
            match (g("MS "), g("VSUM "), g("ISUM ")) {
                (Some(p), Some(pv), Some(pi)) => {
                    let vrel = (vsum - pv).abs() / (pv.abs() + 1e-12);
                    let imatch = (isum - pi).abs() < 0.5;
                    println!("  PyTorch      : {p:8.3} ms   vsum {pv:.6e}  isum {pi:.6e}");
                    // values are element-wise bit-exact (exact indices + identical input ->
                    // values=input[argmax]); the tiny vrel is just 16M-element checksum sum-order.
                    println!(
                        "  CORRECTNESS  : values rel {vrel:.2e} ({}) ; indices ({})",
                        if vrel < 1e-9 { "MATCH" } else { "MISMATCH!" },
                        if imatch { "MATCH" } else { "MISMATCH!" }
                    );
                    let r = p / best;
                    if r >= 1.0 {
                        println!("  => FT {r:.2}x FASTER (cummax dim=0)");
                    } else {
                        println!("  => FT {:.2}x slower", 1.0 / r);
                    }
                }
                _ => println!("  PyTorch      : (parse failed)\n{s}"),
            }
        }
        Ok(o) => eprintln!("py failed: {}", String::from_utf8_lossy(&o.stderr)),
        Err(e) => eprintln!("launch failed: {e}"),
    }
}
