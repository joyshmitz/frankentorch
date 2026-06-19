# frankentorch-kgs4.129 max_pool3d fused-sum rejection

Agent: IvoryDeer (`cod-b`)
Date: 2026-06-19
Host: `thinkstation1`
PyTorch oracle: `/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python`, PyTorch `2.12.1+cpu`
Target: `gauntlet_max_pool3d_grad` (`N=2,C=32,D=16,H=32,W=32`, f64, `kernel=2`, `stride=2`, `loss=sum(out)`)

## Baseline

Command:

```bash
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool3d --noplot
```

Headline:

- FrankenTorch median: `16.692 ms`
- PyTorch median: `2.3587 ms`
- Ratio: FrankenTorch `7.08x` slower

Stage routing rows:

- setup tensor: `240.66 us`
- forward-only: `3.8604 ms`
- sum-only: `1.0497 ms`
- backward-only: `8.5067 ms`
- raw kernel forward+indices: `809.89 us`
- raw kernel backward-from-indices: `1.7109 ms`

## Candidate

Lever tried and reverted:

- Tag f64 max_pool3d custom autograd nodes.
- Rewrite `tensor_sum(max_pool3d_output)` into a fused scalar-output tape node when the max_pool3d intermediate had no retain-grad or hook observer.
- Scatter the scalar upstream gradient through the saved argmax sidecar directly to the original input, avoiding the dense all-ones `dout` buffer.

Same public benchmark call site:

```rust
let out = session.functional_max_pool3d(x, (2, 2, 2), (2, 2, 2))?;
let loss = session.tensor_sum(out)?;
session.tensor_backward(loss)?;
```

Candidate headline:

- FrankenTorch median: `21.590 ms`
- PyTorch median: `1.9601 ms`
- Ratio: FrankenTorch `11.01x` slower
- Delta vs baseline FrankenTorch median: `1.29x` slower

Candidate stage rows:

- setup tensor: `226.88 us`
- forward-only: `4.2581 ms`
- sum-only: `1.2627 ms`
- backward-only: `7.9496 ms`
- raw kernel forward+indices: `759.65 us`
- raw kernel backward-from-indices: `1.6929 ms`

## Verdict

Negative-evidence ledger:

- Wins: `0`
- Losses: `1`
- Neutral: `0`

The fused scalar tape node slightly reduced the isolated backward row but made
the full public PyTorch-vs-FrankenTorch row materially worse. The API/tape
rewrite is rejected and product source was reverted. Do not retry this exact
fused scalar-output node or the earlier standalone unit-`dout` scatter route.

Next route should decompose the `8 ms` API backward overhead without changing
the public graph semantics first: gradient-report cloning, persistent-gradient
materialization, dependency scheduling, and retained intermediate gradient
surface are stronger suspects than raw max_pool3d scatter.

## rch note

`rch exec -- cargo bench ... max_pool3d --noplot` built and ran the
FrankenTorch row on `hz1`, but the worker lacked `torch`, so it could not
produce head-to-head ratio evidence. That remote result was treated as
environment evidence only; the local PyTorch CPU venv rows above are the
keep/reject proof.
