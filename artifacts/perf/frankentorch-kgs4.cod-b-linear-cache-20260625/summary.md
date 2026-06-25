# frankentorch-kgs4 cod-b persistent packed f64 linear cache rejection

Agent: PearlReef
Date: 2026-06-25

## Candidate

Port the old `ivorydeer/kgs4-56-duplicate-keep-evidence` idea at kernel level:
pack row-major f64 Linear weights `[out,in]` once into a contiguous `[in,out]`
matrix, then reuse `dgemm` instead of calling the current transposed-weight
`dgemm_bt` path.

This tests the only plausible successor to the rejected per-call B-panel
packing family: persistent packing amortized across repeated Linear calls.

## Behavior Proof While Candidate Was Present

Command:

```bash
AGENT_NAME=PearlReef \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
cargo test -p ft-kernel-cpu packed_linear_weight_f64_matches_linear_tensor_bit_exact \
  --lib -- --nocapture
```

Result: passed, `1 passed; 0 failed; 549 filtered out`.

The test compared the packed-weight path against `linear_tensor_f64` with and
without bias across representative shapes and required `to_bits()` equality.

## Head-To-Head Measurement While Candidate Was Present

Command:

```bash
AGENT_NAME=PearlReef \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
FT_TORCH_THREADS=32 \
FT_TORCH_INTEROP_THREADS=32 \
cargo run --release -p ft-kernel-cpu --example gemm_bt_ab
```

Measured rows:

| shape | current FT `dgemm_bt` | candidate packed FT | PyTorch | candidate vs current | candidate vs PyTorch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[512,1024] @ W[1024,1024]^T` | 3.224 ms | 3.844 ms | 2.041 ms | 1.19x slower | 1.88x slower |
| `[1024,1024] @ W[1024,1024]^T` | 4.988 ms | 5.407 ms | 4.095 ms | 1.08x slower | 1.32x slower |
| `[2048,512] @ W[2048,512]^T` | 10.881 ms | 11.045 ms | 12.619 ms | 1.02x slower | 1.14x faster |

The large-row shape is already a PyTorch win on the shipped current path
(`10.881 ms` vs `12.619 ms`, FT 1.16x faster). The packed candidate narrows that
win and loses to current FT, so it is not causal win evidence.

## Decision

REVERTED. No source retained.

Persistent f64 Linear weight packing is not a keepable kernel-level lever on the
current `dgemm_bt` implementation. Do not retry this family unless a higher
level API cache proves construction/reuse wins end-to-end on a workload that
actually calls the cached route and beats the current shipped kernel.

## Final Verification After Revert

Command:

```bash
AGENT_NAME=PearlReef \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
cargo test -p ft-conformance
```

Result: passed, including `ft_conformance` lib tests, bin unit tests,
integration tests, smoke tests, and doc tests.
