# frankentorch-2rsa6 closeout - saved-stats f64 group_norm grad route

## Decision

REJECT.

The candidate preserved behavior but did not produce a material win. The source
change was removed; no group_norm source change is kept for this bead.

## Profile-backed target

- Source profile: `artifacts/perf/frankentorch-next-reprofile-20260617/current_top_train_reprofile.log`
- Hot row: `group_norm/grad_32x256x28x28` `[482.72 ms 493.98 ms 505.54 ms]`
- Dedicated local baseline:
  `artifacts/perf/frankentorch-2rsa6/pass1_local_baseline_group_norm_grad.log`
  reported `[502.78 ms 509.77 ms 516.70 ms]`
- Baseline artifact SHA-256:
  `a0695e47c8019687ab48e8ec947c7c1e2430930b5d07301f6f3d1ce5b10689a1`

## One lever

The rejected candidate added saved per-`(batch, group)` mean/rstd sidecars in
f64 group_norm forward and reused them in backward to avoid stats rescans.

## Isomorphism proof

- Output order preserved: forward still wrote each group row in ascending element
  order.
- Floating point preserved: mean/rstd computations and backward formulas matched
  the recomputing route bit-for-bit in the kernel proof.
- Affine gradient order preserved: dweight/dbias accumulation stayed in serial
  group/element order.
- RNG preserved: no RNG is involved.
- Kernel proof:
  `artifacts/perf/frankentorch-2rsa6/pass2_kernel_proof_group_norm_saved_stats.log`,
  SHA-256 `13cd7ca531757eba9adce87555b83c12f99045f82c8bdb4a14fa0a29af16e923`
- API proof:
  `artifacts/perf/frankentorch-2rsa6/pass2_api_proof_group_norm_grad.log`,
  SHA-256 `c77988083523f52d53ce9f3b673241900d183833a208c659f85b075b7259fb56`

## Rebench

Final local Criterion rebench:

- Artifact: `artifacts/perf/frankentorch-2rsa6/pass3_local_rebench_group_norm_grad.log`
- Candidate: `[498.68 ms 507.61 ms 515.85 ms]`
- Baseline median to candidate median: `509.77 ms -> 507.61 ms`
- Criterion result: change `[-2.4379% -0.4240% +1.6957%]`, `p = 0.72`

Score:

- Impact: `0.00`
- Confidence: `0.98`
- Effort: `0.60`
- Score: `0.00`

## Outcome

No source change was kept. The next pass should avoid saved-stats normalization
micro-levers and move to a different profile-backed primitive.
