# frankentorch-fdn1v rejection: direct reduced Gaussian NLL grad

## Target

Profile-backed hotspot from broad `ops_bench` routing:

- `gaussian_nll/grad_8m`: broad median about `836.65 ms`
- focused baseline on `vmi1149989`: `806.19 ms .. 829.27 ms .. 852.70 ms`

## Lever Tested

For same-shape f64 `tensor_gaussian_nll_loss(..., reduction in {"mean", "sum"})`
with gradients, return the reduced scalar directly and compute
`input`/`target`/`var` gradients from the scalar upstream gradient.

The lever avoided materializing the per-element loss tensor and avoided
`tensor_mean(per)` backward expansion.

## Isomorphism Proof

Ordering/tie-breaking/RNG:

- No RNG calls were changed.
- `reduction="none"`, broadcast, non-f64, and fallback paths were unchanged.
- Reduced forward used the same element formula and the same pairwise split tree
  as `gaussian_nll_forward_f64` followed by `sum_tensor_contiguous_f64`.
- `mean` backward used the same `grad_scalar * (1 / n)` scaling as
  `TensorNodeOp::Mean`.

Golden evidence:

- API direct reduced path matched the old `none -> mean/sum` path bit-for-bit
  for output, `dx`, `dt`, and `dv`, `full=false/true`, `mean/sum`.
- Golden FNV digest: `0xfb1a068bd0ae061f`.
- `pass3_api_golden_gaussian_nll_reduced_grad.log`: 1 focused API test passed.
- `pass4_kernel_gaussian_nll_reduced_bits.log`: 2 focused kernel tests passed.

## Benchmark Result

Same worker: `vmi1149989`

- Before: `gaussian_nll/grad_8m [806.19 ms 829.27 ms 852.70 ms]`
- After: `gaussian_nll/grad_8m [978.36 ms 1.0274 s 1.0702 s]`

Result: regression, `0.807x` median speedup. Score is below `2.0`.

## Decision

Rejected and source reverted. The tested scalar-reduced autograd function is not
the right primitive for this hotspot; likely cost shifted to the new custom
function and three full gradient vector writes without enough savings from
skipping the per-element loss node. Next pass should route away from this lever
family toward a different profiled hotspot or a deeper loss-gradient primitive.
