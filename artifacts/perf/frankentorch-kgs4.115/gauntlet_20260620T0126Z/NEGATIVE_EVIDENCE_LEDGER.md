# frankentorch-kgs4.115 Negative-Evidence Ledger

Agent: `IvoryDeer`

## Workload

- Bead: `frankentorch-kgs4.115`
- Operation: f32 GroupNorm forward plus backward with all-ones `dy`
- Shape: `[N,C,H,W]=[8,64,28,28]`, `num_groups=32`, affine weight and bias
- Loss: `sum(group_norm(x, weight, bias))`
- Rust benchmark: `cargo run --release -p ft-api --example group_norm_f32_grad_ab`
- PyTorch oracle: local CPU torch `2.12.1+cpu` from `/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python`

## Result

- Same-worker rch current on `hz1`: composed `101.96 ms`, fused current `11.72 ms`, composed/fused speedup `8.70x`.
- Same-worker rch parent baseline on `hz1` at `e1927d48`: fused `19.13 ms`.
- Current/parent latency ratio: `0.6126x`; the current f32 unit-dy path is `1.63x` faster than parent.
- PyTorch best-of-12 local CPU: `0.615446 ms` best, `0.989997 ms` median.
- Current FrankenTorch best vs PyTorch best: `19.04x` slower.
- Win/loss/neutral vs PyTorch: `0W / 1L / 0N`.

## Verdict

Keep the already-landed f32 GroupNorm all-ones-`dy` backward path as a measured
internal win. It removes a real regression versus the parent implementation, but
it does not close the upstream gap. No source revert.

Do not retry another narrow `dy == 1` micro-specialization for this f32 GroupNorm
shape unless a new profile shows the primitive itself is again dominant after
session setup, tape allocation, tensor materialization, and scalar-sum backward
have been separated. The remaining PyTorch gap should route to deeper levers:
arena-backed tensor/tape allocation, fused training primitives, persistent
workspaces, parallel f32 kernel scheduling, or an explicit f32 Criterion/PyTorch
gauntlet row for `[32,256,28,28]`.

## Evidence

- `current_rch_group_norm_f32_ab.log`
- `baseline_parent_rch_group_norm_f32_ab.log`
- `local_pytorch_group_norm_f32_grad.log`
- `test_ft_kernel_cpu_group_norm_f32_unit_dy.log`
- `test_ft_api_group_norm_f32_grad.log`
- `test_ft_conformance_strict_scheduler.log`
