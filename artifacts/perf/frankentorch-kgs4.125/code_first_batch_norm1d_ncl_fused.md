# frankentorch-kgs4.125 - native NCL BatchNorm1d fused path

Agent: IvoryDeer / cod-a
Date: 2026-06-19; measured closeout 2026-06-20
Status: measured keep; still loses to PyTorch

## Workload Trigger

BatchNorm remains a realistic train-trace hotspot in the current no-gaps profile:

- `artifacts/perf/frankentorch-next-reprofile-20260617c/current_top_train_reprofile_after_16m8a.log`
- `batch_norm/grad_1d_8192x1024` remains a first-tier row.
- Conv1d/sequence models commonly feed BatchNorm1d as native `[N, C, L]`.

Measured closeout is recorded in
`artifacts/perf/frankentorch-kgs4.125/gauntlet_20260620T1336Z/SUMMARY.md`.

## Lever

`functional_batch_norm1d` accepted 3-D `[N, C, L]` by:

1. permuting to `[N, L, C]`,
2. reshaping to `[N*L, C]`,
3. calling the 2-D BatchNorm path,
4. reshaping back to `[N, L, C]`,
5. permuting back to `[N, C, L]`.

The fused BatchNorm helper and CPU kernels already support native
`[batch, channels, spatial]` layout. This patch tries
`try_fused_batch_norm(input, spatial=L)` first for 3-D inputs and preserves the
old fold path for unsupported/mixed cases.

Alien mapping:

- Cache-aware layout: keep native NCL and avoid two layout transforms.
- Communication avoidance: remove intermediate tape traffic before the fused
  kernel.
- One-lever rule: no BatchNorm math or reduction order changes.

## Correctness Guard

Added `functional_batch_norm1d_3d_native_fused_matches_fold_reference_bits`.

The guard builds two graphs:

- candidate: direct `functional_batch_norm1d([N,C,L])`, which takes the new
  fused path when eligible;
- reference: explicit old fold route
  `[N,C,L] -> permute [N,L,C] -> reshape [N*L,C] -> functional_batch_norm1d`
  -> reshape -> permute back.

It compares output, updated running mean/var, input grad, weight grad, and bias
grad bit-for-bit.

## Negative-Evidence Ledger

| Attempt | Evidence | Decision |
| --- | --- | --- |
| BatchNorm1d f64 spatial=1 unit-dy branch | `artifacts/perf/frankentorch-etebu/closeout_batch_norm1d_unit_dy_reject.md`; no significant win. | Do not retry f64 spatial=1 unit-gradient kernel micro-specialization. |
| BatchNorm1d row-major stats scan | `artifacts/perf/frankentorch-2i1cq/closeout_batch_norm1d_row_major_stats_reject.md`; candidate regressed. | Do not retry stats-order/layout scan changes. |
| BatchNorm2d f64 unit-dy branch | `artifacts/perf/frankentorch-6olvt/closeout_batch_norm2d_unit_dy_keep.md`; modest keep in different spatial workload. | Positive adjacent evidence only; this pass is API/tape layout removal for NCL. |
| f32 BatchNorm unit-dy | `artifacts/perf/frankentorch-kgs4.114/code_first_f32_batch_norm_unit_dy.md`; code-first pending. | Separate dtype/kernel lever; do not conflate with NCL layout removal. |

The batch Criterion and conformance closeout kept the native NCL route. Do not
retry BatchNorm1d NCL fold-removal itself. Remaining work should target f64
scalar-loss fusion, dense all-ones `dy` removal, tape/workspace reuse, saved
stat reuse, or a PyTorch-parity proof for zero BatchNorm sum-loss input grads.

## Verification

Original code-first gate:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b cargo check -p ft-api
```

Result: PASS on 2026-06-19.

Measured closeout:

- RCH Criterion on `vmi1227854`: native NCL median `4.3741 ms`,
  fold-reference median `30.484 ms`; native `6.97x` faster.
- Local same-host row coarsening: native `11.865 ms -> 10.914 ms`, `1.09x`
  faster.
- Local PyTorch CPU median `2.251326 ms`; final FT/PyTorch ratio `4.85x`
  slower.
- Focused BatchNorm kernel tests passed.
- Native NCL vs explicit fold bit guard passed before and after row coarsening.
- `ft-api --benches` check passed.
- `ft-kernel-cpu --lib` check and clippy passed.
- `ft-api --bench ops_bench` clippy passed after fixing two pre-existing
  single-element loops in the bench file, then passed again after the UBS
  label-comparison cleanup.
- Full `ft-conformance` passed via local fallback after RCH had no admissible
  workers.
- Scoped UBS rerun reported 0 critical findings; the remaining warning
  inventory is the existing broad bench/kernel surface.
