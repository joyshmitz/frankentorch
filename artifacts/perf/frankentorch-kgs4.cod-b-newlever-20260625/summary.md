# frankentorch-kgs4 cod-b new-lever pass

Agent: PearlReef
Date: 2026-06-25
Target: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b`

## Worktree Scan

`git worktree list --porcelain` plus an `origin/main..HEAD` count found no
clean unlanded worktree win. The only ahead worktree was
`/data/projects/frankentorch-gxpb2-pass10`, commit `1e0ec9bc`, an explicit
large-n row-SIMD rejection.

## Baselines

Current-main h2h probes:

```bash
AGENT_NAME=PearlReef \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
cargo run --release -p ft-api --example masked_select_h2h
```

Result: masked_select FT `38.52 ms`, PyTorch `30.87 ms`, FT `1.25x slower`,
checksum `MATCH`.

```bash
AGENT_NAME=PearlReef \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
cargo run --release -p ft-api --example cdist_p1_headtohead
```

Initial cdist p=1 run: FT `20.433 ms`, PyTorch `10.099 ms`, FT `2.02x
slower`. A scalar control rerun after reverting the candidate path measured FT
`15.935 ms`, PyTorch `7.944 ms`, FT `2.01x slower`, so the first FT baseline
was treated as contention/noise.

## Candidate

The old cdist p=1 ledger note allowed only a deeper SIMD/tiled Manhattan retry.
The candidate tiled the p=1 fused `cdist_forward_f64` output columns:

- 4-column safe `wide::f64x4` tile: FT `15.753 ms`, PyTorch `8.764 ms`, FT
  `1.80x slower`.
- 8-column two-accumulator tile: FT `14.879 ms`, PyTorch `9.019 ms`, FT
  `1.65x slower`.
- 8-column confirmation: FT `15.229 ms`, PyTorch `8.789 ms`, FT `1.73x
  slower`.

Each output cell preserved the original left-to-right `k` accumulation order;
only independent output columns were interleaved.

## Proof

- `cargo check -p ft-kernel-cpu --all-targets`: pass for the 4-column
  candidate.
- `cargo check -p ft-kernel-cpu --all-targets`: pass for the 8-column
  candidate.
- `cargo test -p ft-api cdist --lib -- --nocapture`: `12 passed`.
- After source restore, `cargo test -p ft-conformance`: pass.

## Decision

REVERTED/no source retained. The best candidate reduced the scalar-control FT
time by only about 4-7% and still lost to PyTorch by `1.65-1.73x`. Do not retry
`cdist p=1` column tiling or `wide::f64x4` interleaving without a lower-level
profile proving a new bottleneck.
