# f32 batched eigvalsh head-to-head scorecard

Date: 2026-06-21
Agent: IvoryDeer / cod-b
Parent validated: `fc3b2dcb5909cb519ee78840c72e652a6838b891`
Lane: native f32 batched `linalg.eigvalsh` for contiguous `[..., k, k]` no-grad tensors.

## Lever

`origin/main` already contains the production lever at validation time: `eigvalsh_batched_contiguous_f32` in `ft-kernel-cpu` and the f32 batched no-grad API route in `ft-api`.

The radical lever is the f32 mirror of the previously kept f64 batched linalg pattern: keep the batch flattened and run independent small eigensolves in parallel, avoiding both PyTorch's per-plane LAPACK loop overhead and FrankenTorch's older f32->f64->f32 fallback.

Skill mapping:
- alien graveyard: vectorized execution/manual flattening plus numerical linear algebra specialization.
- alien artifact: bit-exact-by-construction against looping native 2-D f32 eigvalsh.
- extreme optimization/profiling: same-worker before/after timing, then PyTorch head-to-head.

## Same-worker FrankenTorch A/B

Worker: `hz2`
Command family: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo run --release -p ft-api --example batched_eigvalsh_f32_h2h`

| Shape | Fallback f32->f64->f32 | Native f32 batched | Internal speedup |
|---|---:|---:|---:|
| `[100000,4,4]` | `14.0 ms` | `6.9 ms` | `2.03x` |
| `[20000,16,16]` | `51.4 ms` | `13.3 ms` | `3.86x` |
| `[4000,32,32]` | `24.6 ms` | `14.1 ms` | `1.74x` |

## PyTorch comparator

Remote RCH workers lacked `torch`, so PyTorch was measured with the local CPU sidecar:
`/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python`, `torch 2.12.1+cpu`, `torch.set_num_threads(8)`.

| Shape | FrankenTorch native | PyTorch CPU | Ratio |
|---|---:|---:|---:|
| `[100000,4,4]` | `6.9 ms` | `50.496331 ms` | `7.32x` faster |
| `[20000,16,16]` | `13.3 ms` | `99.272275 ms` | `7.46x` faster |
| `[4000,32,32]` | `14.1 ms` | `70.954784 ms` | `5.03x` faster |

Head-to-head score for this pass: `3W / 0L / 0N` vs PyTorch.

Checksum sums matched the expected f32 rounded totals:
- `[100000,4,4]`: FT `1.7920e6`, PyTorch `1.792000188184e6`
- `[20000,16,16]`: FT `5.2736e6`, PyTorch `5.273598919497e6`
- `[4000,32,32]`: FT `4.1574e6`, PyTorch `4.157439016443e6`

## Gates

Green:
- `rch exec -- cargo test -p ft-kernel-cpu eigvalsh_batched_f32_matches_looping_2d_bit_exact --profile release`
- `rch exec -- cargo test -p ft-api eigvalsh_batched_f32_native_keeps_shape_and_dtype --profile release`
- `rch exec -- cargo test -p ft-conformance --profile release`
- `rustfmt --edition 2024 --check crates/ft-api/examples/batched_eigvalsh_f32_h2h.rs`
- `rch exec -- cargo check -p ft-api --example batched_eigvalsh_f32_h2h`
- `ubs crates/ft-api/examples/batched_eigvalsh_f32_h2h.rs artifacts/.../SCORECARD.md` exited `0`; zero critical findings. Remaining contextual warnings are benchmark-harness patterns: FT call `unwrap()`, shape-proven direct indexing, per-repetition input clone, and loop-local formatting.

Known non-green repo-wide gates not introduced by this lane:
- `cargo fmt --check` reports broad pre-existing formatting drift across `ft-api` examples and large source files.
- `cargo clippy -p ft-kernel-cpu --lib -- -D warnings` reports existing `clippy::manual_memcpy` warnings in scan helpers.

## Verdict

Keep. This is a measured PyTorch win and a measured internal improvement. Remaining batched-linalg gaps are f32 `svdvals`, f32 `qr`, and tiny-k `svd` class probes.
