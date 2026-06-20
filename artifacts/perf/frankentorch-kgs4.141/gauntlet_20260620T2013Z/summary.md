# frankentorch-kgs4.141 summary

## Verdict

Rejected and reverted. The f32 BatchNorm2d scalar-loss algebraic-zero backward
candidate did not produce a statistically significant end-to-end speedup.

## Workload

- FrankenTorch row: `gauntlet_batch_norm2d_f32_grad/frankentorch_kgs4_136_scalar_sum`
- Shape: `[N,C,H,W]=[32,256,28,28]`
- Operation: training BatchNorm2d f32, affine weight/bias require gradients,
  scalar loss via `functional_batch_norm2d_sum(...).backward()`
- Candidate: make `batch_norm_backward_scalar_f32` return `dx = 0`,
  `dweight = 0`, `dbias = upstream * batch * spatial`

## Measurements

- RCH baseline on `vmi1227854`:
  - materialized row median `118.47 ms`
  - scalar-sum row median `75.361 ms`
  - PyTorch arm failed on the worker: `ModuleNotFoundError: No module named 'torch'`
- RCH candidate after-run on `vmi1293453`:
  - materialized row median `348.35 ms`
  - scalar-sum row median `181.90 ms`
  - PyTorch arm failed on the worker: `ModuleNotFoundError: No module named 'torch'`
  - This is not a keep/reject comparison because the unchanged materialized row
    was `2.94x` slower than the RCH baseline worker.
- Local paired fallback:
  - exact requested target dir `/data/projects/.rch-targets/frankentorch-cod-a`
    was not used locally because it contained artifacts from a different nightly
    and failed with `E0514`; no cleanup was performed.
  - paired local target: `/data/projects/.rch-targets/frankentorch-cod-a-local-pair`
  - baseline median `116.70 ms`
  - candidate median `115.48 ms`
  - Criterion change `[-3.2139%, -1.0450%, +1.1755%]`, `p = 0.40`
  - Result: `No change in performance detected.`
- Local PyTorch comparator:
  - PyTorch `2.12.1+cpu`, 32 threads
  - five 40-iteration totals: `0.298917255015`, `0.298686239053`,
    `0.266571774962`, `0.279750167974`, `0.315121468971` seconds
  - median per iteration: `7.467156 ms`
  - candidate scalar-sum ratio vs PyTorch: `15.46x` slower

## Correctness and Gates

- Candidate kernel scalar tests passed after contract update:
  `cargo test -p ft-kernel-cpu batch_norm_f32_scalar --lib --profile release`
- Candidate API scalar BatchNorm2d test first failed under the old
  materialized-residue contract (`dx[0]: scalar 0 vs materialized -1.8479706e-7`)
  and passed after a temporary product-zero contract update.
- Product source and temporary test-contract changes were reverted because the
  measured speedup was neutral.
- Reverted-tree conformance passed:
  `rch exec -- cargo test -p ft-conformance --profile release`
  with all selected unit/bin/integration/doc tests green.

## Follow-Up Direction

Do not retry this simple f32 scalar-backward algebraic-zero lever by itself.
The end-to-end scalar-sum path is dominated elsewhere. Route the remaining gap
to output deforestation, saved-stat/workspace reuse across forward/backward,
session/tape arena allocation, generated shape-specialized scalar-loss kernels,
or a gradient representation that avoids allocating and zero-filling the large
`dx` buffer when the product gradient is known zero.
