use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn main() {
    for (bb, k) in [(100_000usize, 4usize), (20_000, 16), (4_000, 32)] {
        let mut data = vec![0.0f32; bb * k * k];
        for b in 0..bb {
            for i in 0..k {
                for j in 0..k {
                    data[b * k * k + i * k + j] = (((b * 7 + i * 13 + j * 5) % 97) as f32) * 0.01;
                }
            }
        }
        for b in 0..bb {
            for i in 0..k {
                for j in 0..k {
                    let s = (data[b * k * k + i * k + j] + data[b * k * k + j * k + i]) * 0.5;
                    data[b * k * k + i * k + j] = s;
                }
            }
            for i in 0..k {
                data[b * k * k + i * k + i] += k as f32;
            }
        }

        let mut best = f64::INFINITY;
        let mut esum = 0.0f64;
        for _ in 0..5 {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session
                .tensor_variable_f32(data.clone(), vec![bb, k, k], false)
                .unwrap();
            let start = Instant::now();
            let evals = session.tensor_linalg_eigvalsh(x).unwrap();
            let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
            if elapsed_ms < best {
                best = elapsed_ms;
                esum = session
                    .tensor_values_f32(evals)
                    .unwrap()
                    .iter()
                    .map(|&v| v as f64)
                    .sum();
            }
        }

        let pysrc = format!(
            r#"
import time, torch
torch.set_num_threads(8)
B,k={bb},{k}
idx=torch.arange(B*k*k)
b=idx//(k*k)
r=(idx//k)%k
c=idx%k
d=(((b*7+r*13+c*5)%97).double())*0.01
A=d.reshape(B,k,k)
A=(A+A.transpose(-1,-2))*0.5
A=(A+k*torch.eye(k,dtype=torch.float64)).float()
for _ in range(2):
    torch.linalg.eigvalsh(A)
ts=[]
for _ in range(5):
    t=time.perf_counter()
    torch.linalg.eigvalsh(A)
    ts.append((time.perf_counter()-t)*1e3)
w=torch.linalg.eigvalsh(A)
print("MS", sorted(ts)[0])
print("ESUM", w.double().sum().item())
"#
        );
        let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
        print!("B={bb} k={k} f32 eigvalsh: FT {best:.1} ms  esum {esum:.4e}");
        let mut child = Command::new(&python)
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        if let Ok(child) = child.as_mut()
            && let Some(stdin) = child.stdin.as_mut()
        {
            let _ = stdin.write_all(pysrc.as_bytes());
        }
        if let Ok(output) = child.and_then(|child| child.wait_with_output()) {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let get = |prefix: &str| {
                    stdout
                        .lines()
                        .find_map(|line| line.strip_prefix(prefix))
                        .and_then(|value| value.trim().parse::<f64>().ok())
                };
                if let (Some(torch_ms), Some(torch_esum)) = (get("MS "), get("ESUM ")) {
                    let rel = (esum - torch_esum).abs() / (torch_esum.abs() + 1e-6);
                    let ratio = torch_ms / best;
                    let verdict = if ratio >= 1.0 {
                        format!("FT {ratio:.2}x FASTER")
                    } else {
                        format!("FT {:.2}x slower", 1.0 / ratio)
                    };
                    println!(
                        "  | PyTorch {torch_ms:.1} ms  esum {torch_esum:.4e} | {} | {verdict}",
                        if rel < 1e-4 { "MATCH" } else { "DIFF" }
                    );
                }
            } else {
                println!();
                eprintln!("{}", String::from_utf8_lossy(&output.stderr));
            }
        } else {
            println!();
        }
    }
}
