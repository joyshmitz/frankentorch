# frankentorch-kgs4.150 scorecard: f32 SDPA unit-dout backward

## Lever

Specialize first-order f32 SDPA backward when the upstream gradient is exactly
all ones, the common `sum(attn)` scalar-loss case used by the head-to-head
training-step gauntlet. The kernel is algebraically identical to dense
`dout = ones`, but avoids materializing `dout` and replaces the all-ones
`dP`/`dV` products with reductions:

- `dP[i,j] = sum_l V[j,l]`
- `dV[j,l] = sum_i P[i,j]`

`dQ` and `dK` keep the existing GEMM path.

## Measurements

All FrankenTorch builds used
`AGENT_NAME=cod-a CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a`
with per-crate `rch exec -- cargo run --release -p ft-api --example sdpa_f32_headtohead`.
Workers lack torch, so PyTorch ratios come from the retrieved `vmi1227854`
release binary plus the local CPU torch sidecar
`/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python`.

| Row | Current FT | Candidate FT | Delta |
| --- | ---: | ---: | ---: |
| RCH `vmi1227854` non-causal | 31.363 ms | 28.152 ms | 1.114x faster |
| RCH `vmi1227854` causal | 27.410 ms | 22.669 ms | 1.209x faster |
| local sidecar non-causal | 35.281 ms | 33.696 ms | 1.047x faster |
| local sidecar causal | 33.647 ms | 31.512 ms | 1.068x faster |

| Row | Candidate FT | PyTorch sidecar | Ratio |
| --- | ---: | ---: | ---: |
| non-causal | 33.696 ms | 19.623 ms | FT 1.72x slower |
| causal | 31.512 ms | 19.840 ms | FT 1.59x slower |

Score vs PyTorch: `0W / 2L / 0N`.

Internal FT score vs current source: `2W / 0L / 0N` on same-machine PyTorch
sidecar evidence, with stronger same-worker FT-only RCH movement.

## Gates

- `rch exec -- cargo test --profile release -p ft-kernel-cpu sdpa_backward_f32_unit_dout_matches_dense_ones --lib -- --nocapture`: passed.
- `rch exec -- cargo test --profile release -p ft-api sdpa_f32_grad_matches_f64_path --lib -- --nocapture`: passed.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -j 4 -p ft-conformance --profile release`: passed on `vmi1227854` (library, bins, integration, smoke, doctests).

The first `ft-conformance` attempt fell back to local execution and hit mixed
rustc artifacts in the warm target directory; it was discarded as environment
noise and replaced by the remote-required run above.

## Decision

KEEP as a narrow f32 SDPA scalar-loss improvement, not as a PyTorch domination
claim. The ratio ledger remains negative because PyTorch still wins both rows.

Next lever should not be another wrapper around dense upstream gradients. The
remaining gap points at a deeper CPU attention primitive: online softmax plus
fused value accumulation for backward, vectorized exp/softmax within tolerance,
or a BLAS/SIMD-class f32 matmul/attention tile change.
