# BOLD-VERIFY PyTorch head-to-head pass

Agent: `PearlReef`
Date: 2026-06-25
Target dir: `/data/projects/.rch-targets/frankentorch-cod-b`
PyTorch sidecar: `/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python`

## Kept

- `tensor_combinations(input, r=2, with_replacement=false)`, fixture
  `torch.arange(3000, dtype=float64)`, no-grad, 5-iteration minimum:
  FrankenTorch `136.69 ms`, PyTorch `182.20 ms`, checksum match
  `1.349100e10`, FT `1.33x FASTER`.

## Rejected / no source retained

- `qr_grad_h2h`: `k=4` FT `1.09x` slower, `k=16` FT `1.01x` slower.
- `logsumexp` no-grad output-lane fast path present in the dirty tree:
  FT `67.37 ms`, PyTorch `10.56 ms`, FT `6.38x` slower; reverted.
- `masked_select` current baseline: FT `38.50 ms`, PyTorch `29.21 ms`,
  FT `1.32x` slower. Direct no-grad variants measured `44.03 ms` and
  `47.08 ms`, both slower; reverted.
