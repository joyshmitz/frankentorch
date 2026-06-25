# dgemm_bt Panel-Pack Rejections

Agent: PearlReef
Date: 2026-06-25
Target: `dgemm_bt` in `ft-kernel-cpu`

## Worktree Scan

Fresh ancestry check found no clean unlanded worktree win ahead of `main`.
The only ahead worktree was `/data/projects/frankentorch-gxpb2-pass10`, whose
single commit is an explicit large-n row-SIMD rejection.

## Lever

Alien route: cache-aware vectorized execution / panel packing. The candidate
changed the 2-D transposed-B GEMM path from strided `B^T` reads to one packed
contiguous `[k, bj]` panel per output-column block, then reused that panel for
the row tiles. K remained unsplit and output tiles remained disjoint, so the
intended behavior proof was bit-identical accumulation order.

## Baseline

Command:

```bash
AGENT_NAME=PearlReef CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
cargo run --release -p ft-kernel-cpu --example gemm_bt_ab
```

Rows:

- `[512,1024] @ W[1024,1024]^T`: `4.065 ms`
- `[1024,1024] @ W[1024,1024]^T`: `5.911 ms`
- `[2048,512] @ W[2048,512]^T`: `13.654 ms`

PyTorch gauntlet baseline:

```bash
AGENT_NAME=PearlReef CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
FT_TORCH_THREADS=32 FT_TORCH_INTEROP_THREADS=32 \
cargo bench -p ft-api --bench pytorch_gauntlet_bench -- gauntlet_linear_train_hidden_2048 --noplot
```

- FrankenTorch f64 Linear train median: `7.6681 ms`
- PyTorch CPU median: `40.835 ms`
- Baseline ratio: FT `5.32x faster`, but PyTorch variance was high.

## Candidate

Same kernel command after the panel-pack hunk:

- `[512,1024] @ W[1024,1024]^T`: `4.179 ms`, `1.03x slower`
- `[1024,1024] @ W[1024,1024]^T`: `6.191 ms`, `1.05x slower`
- `[2048,512] @ W[2048,512]^T`: `14.301 ms`, `1.05x slower`

The public PyTorch gauntlet while the candidate was present measured:

- FrankenTorch f64 Linear train median: `6.9660 ms`
- PyTorch CPU median: `12.406 ms`
- Candidate ratio: FT `1.78x faster`

That public row is not causal evidence for this hunk: the Linear forward shape
uses the earlier small-M column split and does not dispatch to
`dgemm_bt_2d_parallel`. The targeted kernel example is the decisive row for
this lever and it regressed all three cases.

## Decision

REVERT/no source retained. Do not retry plain per-column-block packing inside
`dgemm_bt_2d_parallel`; the extra panel copy and nested Rayon structure loses to
matrixmultiply's strided-B path on the measured 2-D GEMM shapes. A future
attempt would need a microkernel-native persistent packed-weight path with a
dispatch gate that proves it is invoked by the target Linear shape.

## Follow-up Candidate: Small-M Column Split Packing

Because f64 Linear forward uses the earlier small-M column split rather than
`dgemm_bt_2d_parallel`, a second one-hunk candidate packed each column-split
block into contiguous `[k, bw]` panels before calling `dgemm_mm`.

Measured command:

```bash
AGENT_NAME=PearlReef CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
FT_TORCH_THREADS=32 FT_TORCH_INTEROP_THREADS=32 \
cargo bench -p ft-api --bench pytorch_gauntlet_bench -- gauntlet_linear_train_hidden_2048 --noplot
```

Candidate FT f64 Linear train median: `6.9871 ms`; Criterion reported
`No change in performance detected` versus the prior gauntlet row
(`p = 0.61`). The candidate was therefore reverted as zero-gain. The terminal
stream showed a PyTorch median near `8.9249 ms`, but the persisted log did not
retain the final PyTorch analysis lines, so the keep/reject decision is based on
the persisted FT no-change gate rather than an unstable PyTorch timing.

Decision: REVERT/no source retained. Do not retry per-call packing in the
small-M column split; it spends memory bandwidth transposing the weight panel
without a measurable end-to-end win. The remaining plausible family is a
persistent packed-weight object whose construction is amortized across calls and
whose dispatch is explicit in the API, not hidden inside `dgemm_bt`.
