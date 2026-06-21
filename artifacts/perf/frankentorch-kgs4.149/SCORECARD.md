# frankentorch-kgs4.149 - SDPA 4-D row-block retune rejection

## Target

Standard 4-D f64 no-grad SDPA head-to-head:
`[B=2,H=8,SEQ=512,D=64]`, `scaled_dot_product_attention(q,k,v).abs().sum()`.

Reason: the existing 3-D gauntlet SDPA lane was favorable to FT, but the standard
4-D transformer layout remains a PyTorch loss.

## Commands

Baseline/candidate builds used the existing target dir and RCH:

```bash
AGENT_NAME=cod-a CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a \
  rch exec -- cargo run --release -p ft-api --example sdpa_4d_headtohead
```

PyTorch was unavailable on the selected RCH workers, so same-machine ratios used the
retrieved release binary and the local torch 2.12.1+cpu sidecar:

```bash
ITERS=15 PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
  /data/projects/.rch-targets/frankentorch-cod-a/release/examples/sdpa_4d_headtohead
```

## Results

| Source | FT ms | PyTorch ms | Ratio | Decision |
| --- | ---: | ---: | ---: | --- |
| Current `BR=64`, run A | 7.423 | 5.923 | FT 1.25x slower | baseline |
| Current `BR=64`, run B | 7.019 | 6.285 | FT 1.12x slower | baseline |
| Candidate `BR=32` | 7.627 | 5.949 | FT 1.28x slower | reject |
| Candidate `BR=128` | 8.050 | 6.478 | FT 1.24x slower | reject |

Remote FT-only sanity lines:

- Current `BR=64` on `vmi1153651`: 20.067ms, PyTorch unavailable.
- Candidate `BR=32` on `ovh-a`: 8.544ms, PyTorch unavailable.
- Candidate `BR=128` on `vmi1152480`: 8.492ms, PyTorch unavailable.

## Outcome

`0W / 2L / 0N`. Both tile candidates were slower than current FT and still
lost to PyTorch. The kernel was reverted to `BR=64`; no source change shipped.

## Retry Condition

Do not retry scalar `BR` tuning for this lane. Reopen only with phase timing that
isolates a deeper bottleneck, such as raw-kernel softmax/exp time, GEMM/SIMD limits,
or API readback overhead. Candidate families: online softmax with fused V accumulation,
vectorized exp within tolerance, or matrixmultiply-class SIMD work.
