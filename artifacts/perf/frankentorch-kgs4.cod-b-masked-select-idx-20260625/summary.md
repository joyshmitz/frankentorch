# masked_select parallel kept-index compaction reject

Agent: `PearlReef`
Date: 2026-06-25
Target dir: `/data/projects/.rch-targets/frankentorch-cod-b`
PyTorch sidecar: `/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python`

## Candidate

Fan out `tensor_masked_select` kept-index construction over Rayon for large
masks, preserving row-major order through indexed parallel collection, then keep
the existing `index_select` gather path.

## Result

Command:

```bash
AGENT_NAME=PearlReef \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
cargo run --release -p ft-api --example masked_select_h2h
```

Fixture: `masked_select(x[4_000_000], x > 0)`, f64 no-grad, 6-iteration
minimum. Output matched PyTorch: kept `2_001_643`, checksum `1.274286e6`.

- Candidate FrankenTorch: `116.36 ms`
- PyTorch: `31.74 ms`
- Ratio: FT `3.67x SLOWER`

Prior serial-index baseline from the 2026-06-25 BOLD-VERIFY artifact was
FrankenTorch `38.50 ms`, PyTorch `29.21 ms`, FT `1.32x SLOWER`.

Decision: reject and restore source. The next plausible family is a fused typed
mask+gather kernel, not another index-list compaction tweak.
