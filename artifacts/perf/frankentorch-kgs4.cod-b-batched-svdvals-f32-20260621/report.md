# f32 batched svdvals streaming upcast

Bead/thread: `frankentorch-kgs4.cod-b-batched-svdvals-f32`
Agent: `IvoryDeer`
Date: 2026-06-21

## Lever

`tensor_linalg_svdvals` previously routed f32 inputs through a whole-batch `tensor_to_dtype(F64)`
allocation before using the batched f64 SVD-values kernel. The kept path streams each f32 plane into
the f64 work buffer consumed by the existing Golub-Reinsch values recurrence, preserving the old
casted semantics while avoiding the intermediate f64 tensor.

## Baseline vs Candidate

Command:

```bash
AGENT_NAME=IvoryDeer RCH_WORKER=vmi1153651 \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
rch exec -- cargo run --release -p ft-api --example batched_svdvals_f32_h2h
```

| Shape | Current cast path | Streaming upcast | Speedup |
|---|---:|---:|---:|
| `[100000,8,4]` | 37.030 ms | 24.626 ms | 1.50x |
| `[20000,16,8]` | 30.989 ms | 20.037 ms | 1.55x |
| `[8000,32,16]` | 56.525 ms | 33.687 ms | 1.68x |

## PyTorch Sidecar

RCH workers lacked torch, so PyTorch was measured by running the retrieved release binary locally with
CPU torch 2.12.1+cpu installed under tmpfs:

```bash
PYTHONPATH=/dev/shm/frankentorch-torch-cpu-20260621-ivorydeer \
/data/projects/.rch-targets/frankentorch-cod-b/release/examples/batched_svdvals_f32_h2h
```

| Shape | FrankenTorch | PyTorch | Ratio | Checksum rel |
|---|---:|---:|---:|---:|
| `[100000,8,4]` | 3.818 ms | 149.411 ms | 39.13x faster | 4.076e-10 |
| `[20000,16,8]` | 2.861 ms | 82.526 ms | 28.84x faster | 1.829e-9 |
| `[8000,32,16]` | 4.698 ms | 115.962 ms | 24.68x faster | 9.274e-9 |

## Verification

- `rch exec -- cargo test --release -p ft-kernel-cpu svdvals_batched_f32_upcast_matches_casted_looping_2d_bit_exact --lib`
- `rch exec -- cargo test --release -p ft-api tensor_linalg_svdvals_batched_f32_matches_casted_f64_path_bit_exact --lib`
- `rch exec -- cargo test --release -p ft-conformance`
- `cargo fmt --check --all`

`cargo fmt --check --all` passed on the live checkout after formatting; the final commit was staged from
clean origin-based blobs to avoid unrelated whole-file rustfmt churn. UBS was attempted on the three touched
code files and stopped after roughly two minutes without findings output.
