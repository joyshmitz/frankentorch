# frankentorch-kgs4.116 negative-evidence ledger

## Verdict

Measured keep with PyTorch loss.

The already-landed LayerNorm all-ones upstream-gradient specialization is a real
same-worker FrankenTorch speedup, but it does not beat PyTorch on the matched
CPU oracle run. Keep the code-first fast path; do not treat this as PyTorch
dominance.

## Workload

- Bead: `frankentorch-kgs4.116`
- Lever: `ft-kernel-cpu` `layer_norm_backward_f64`/`layer_norm_backward_f32`
  exact `dy == 1.0` fast path.
- Official benchmark: `ops_bench` `layer_norm/grad_2048x1024`.
- Shape: f64 `[rows, hidden]=[2048,1024]`, affine weight/bias, scalar
  `sum(out)` loss, backward.
- Supporting f32 diagnostic: `layernorm_f32_grad_ab`, shape `[8192,1024]`.

## Same-Worker Rust A/B

- Current command:
  `RCH_WORKER=hz2 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo bench -p ft-api --bench ops_bench -- layer_norm/grad_2048x1024 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot`
- Current worker: `hz2`.
- Current result: `29.606 ms` Criterion point estimate,
  interval `[27.901 ms, 32.407 ms]`.
- Parent baseline: detached worktree at `2aa782006689245ca1b27496c9261b92871f201c`
  (`1e6af1d8^`), same command, same worker `hz2`.
- Parent result: `90.723 ms` Criterion point estimate,
  interval `[87.459 ms, 92.592 ms]`.
- Internal ratio: current/parent `0.3263x`, or parent/current `3.06x`
  faster.

## PyTorch Head-To-Head

- PyTorch command: local CPU Python oracle in
  `/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python`.
- PyTorch version: `2.12.1+cpu`.
- PyTorch workload: `torch.nn.functional.layer_norm` f64 `[2048,1024]`,
  affine weight/bias, scalar `sum(out)` loss, backward, 32 threads.
- PyTorch result: min `5.949352 ms`, median `8.261743 ms`,
  p95 `10.752482 ms`.
- FrankenTorch comparison: current rch Criterion point estimate `29.606 ms`.
- Ratio vs PyTorch median: `3.58x` slower.
- Ratio vs PyTorch min: `4.98x` slower.
- Remote PyTorch caveat: rch workers still lack `torch`; remote Python probe
  failed with `ModuleNotFoundError: No module named 'torch'`.

## Supporting f32 Diagnostic

`layernorm_f32_grad_ab` on `hz2` reported:

- Composed op graph: `1930.66 ms`.
- Fused LayerNorm fast path: `293.49 ms`.
- Ratio: `6.58x` faster.

This f32 diagnostic is not the official bead benchmark because it uses a
different shape and compares against the composed FrankenTorch graph rather
than parent commit Criterion. It supports the route but does not replace the
same-worker f64 Criterion evidence above.

## Win/Loss/Neutral vs PyTorch

- `0W / 1L / 0N`.

## Do Not Retry As

Do not retry LayerNorm saved-forward-stat rematerialization or another tiny
`dy == 1` normalization-only branch for this row unless a fresh profile proves
the kernel branch, not session setup, tape allocation, tensor materialization,
or scalar-sum backward, is again the dominant gap.

The remaining LayerNorm gap should route to deeper levers: arena-backed tensor
and tape allocation, persistent workspaces for normalization backward, fused
loss/backward primitives, parallel deterministic affine-gradient reduction,
f32-native end-to-end rows, or layout/scheduling work that removes whole-array
passes rather than just one constant-gradient multiply.

## Evidence

- `current_rch_ops_layer_norm_grad.log`
- `baseline_parent_rch_ops_layer_norm_grad.log`
- `current_rch_layernorm_f32_ab.log`
- `local_pytorch_layer_norm_f64_grad.log`
- `remote_python_torch_probe.log`
- `test_ft_kernel_cpu_layer_norm_unit_dy.log`
- `test_ft_api_functional_layer_norm.log`
- `test_ft_conformance_strict_scheduler_retry_hz2.log`
