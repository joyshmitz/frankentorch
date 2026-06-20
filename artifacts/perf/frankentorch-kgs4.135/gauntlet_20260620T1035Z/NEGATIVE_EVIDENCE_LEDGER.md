# frankentorch-kgs4.135 Negative-Evidence Ledger

Date: 2026-06-20

## Lever

Added an affine f32 GroupNorm scalar-loss path:

- `ft_kernel_cpu::group_norm_sum_forward_f32`
- `ft_kernel_cpu::group_norm_backward_scalar_f32`
- `FrankenTorchSession::functional_group_norm_sum`

The path computes `sum(group_norm(input, weight, bias))` directly and backprops
a scalar upstream gradient, avoiding the normalized output buffer, `tensor_sum`
tape node, and dense all-ones `dy` buffer.

## Workload

- Shape: `[N,C,H,W]=[8,64,28,28]`
- Groups: `32`
- DType: f32
- Affine weight and bias require gradients
- Loss: scalar sum

## Results

| Evidence | FrankenTorch | Comparator | Ratio |
|---|---:|---:|---:|
| rch baseline diagnostic on `hz1` | composed `105.01 ms` | existing fused `11.52 ms` | fused is `9.12x` faster |
| rch direct candidate on `ovh-a` | scalar-sum `2.10 ms` | existing fused `8.30 ms` | scalar-sum is `3.96x` faster |
| Criterion on `vmi1167313` | scalar-sum median `8.9874 ms` | materialized median `17.139 ms` | scalar-sum is `1.91x` faster |
| PyTorch fair local oracle | scalar-sum `2.10 ms` | PyTorch best `0.376163 ms` | FrankenTorch is `5.58x` slower |

The first PyTorch probe in `baseline_local_pytorch_group_norm_f32_grad.log`
constructed tensors from Python lists inside each timing iteration and is not
used for the release ratio. The accepted oracle is
`baseline_local_pytorch_group_norm_f32_grad_clone.log`, which prebuilds tensors
and clones/detaches per rep.

## Verdict

Keep as an internal win and release-readiness loss.

- Win/loss/neutral vs PyTorch: `0W / 1L / 0N`.
- Same-family retry stop: do not retry another narrow GroupNorm all-ones-`dy`
  microbranch for this row.
- Next route: automatic scalar-loss fusion, arena/bump tape and tensor
  allocation, persistent f32 storage, dtype-conversion removal, cache-blocked
  affine reductions, and scheduler/layout work with direct PyTorch-ratio proof.

## Evidence

- `baseline_rch_group_norm_f32_ab.log`
- `baseline_local_pytorch_group_norm_f32_grad_clone.log`
- `candidate_rch_group_norm_f32_ab.log`
- `candidate_rch_ops_group_norm_f32_bench.log`
- `test_ft_api_group_norm_f32_sum_after_reapply.log`
- `test_ft_kernel_cpu_group_norm_unit_dy_after_reapply.log`
- `test_ft_conformance_strict_scheduler.log`
- `check_ft_api_all_targets.log`
- `check_ft_kernel_cpu_all_targets.log`
- `clippy_ft_api_lib.log`
- `clippy_ft_kernel_cpu_lib.log`
