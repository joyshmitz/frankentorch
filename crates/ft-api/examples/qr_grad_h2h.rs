use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn main() {
    for (bb, k) in [(100_000usize, 4usize), (20_000usize, 16usize)] {
        let mut a: Vec<f64> = (0..bb * k * k)
            .map(|x| (((x * 2_246_822_519usize) % 9_941) as f64) * 0.002 - 9.0)
            .collect();
        for b in 0..bb {
            for i in 0..k {
                a[b * k * k + i * k + i] += 20.0;
            }
        }

        let mut best = f64::INFINITY;
        let mut err = 0.0;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(a.clone(), vec![bb, k, k], true).unwrap();
            let t = Instant::now();
            let (q, r) = s.tensor_linalg_qr(x, true).unwrap();
            let qq = s.tensor_mul(q, q).unwrap();
            let rr = s.tensor_mul(r, r).unwrap();
            let sq = s.tensor_sum(qq).unwrap();
            let sr = s.tensor_sum(rr).unwrap();
            let loss = s.tensor_add(sq, sr).unwrap();
            let rep = s.tensor_backward(loss).unwrap();
            let g = s.tensor_gradient(&rep, x).unwrap();
            let elapsed = t.elapsed().as_secs_f64() * 1e3;
            if elapsed < best {
                best = elapsed;
                let check_len = bb.min(2_000) * k * k;
                let mut e = 0.0;
                for i in 0..check_len {
                    e += (g[i] - 2.0 * a[i]).abs();
                }
                err = e / check_len as f64;
            }
        }

        let pysrc = format!(
            r#"
import time, torch
torch.set_num_threads(8)
B, k = {bb}, {k}
idx = torch.arange(B*k*k, dtype=torch.float64)
A0 = (((idx * 2246822519) % 9941).double().mul(0.002).sub(9.0)).reshape(B, k, k)
A0 = A0 + 20.0 * torch.eye(k, dtype=torch.float64)
def step():
    A = A0.clone().requires_grad_(True)
    Q, R = torch.linalg.qr(A, mode="reduced")
    ((Q * Q).sum() + (R * R).sum()).backward()
    return A.grad
for _ in range(2):
    step()
ts = []
for _ in range(5):
    A = A0.clone().requires_grad_(True)
    t = time.perf_counter()
    Q, R = torch.linalg.qr(A, mode="reduced")
    ((Q * Q).sum() + (R * R).sum()).backward()
    ts.append((time.perf_counter() - t) * 1e3)
print("MS", sorted(ts)[0])
"#
        );
        let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".into());
        print!("k={k}: FT {best:.1}ms (grad-2A err {err:.1e})");
        if let Ok(output) = Command::new(&python).arg("-c").arg(&pysrc).output() {
            if output.status.success() {
                if let Some(torch_ms) =
                    String::from_utf8_lossy(&output.stdout)
                        .lines()
                        .find_map(|line| {
                            line.strip_prefix("MS ")
                                .and_then(|value| value.trim().parse::<f64>().ok())
                        })
                {
                    let ratio = torch_ms / best;
                    if ratio >= 1.0 {
                        println!(" | torch {torch_ms:.1}ms | FT {ratio:.2}x FASTER");
                    } else {
                        println!(" | torch {torch_ms:.1}ms | FT {:.2}x slower", 1.0 / ratio);
                    }
                }
            } else {
                eprintln!("{}", String::from_utf8_lossy(&output.stderr));
            }
        } else {
            println!();
        }
    }
}
