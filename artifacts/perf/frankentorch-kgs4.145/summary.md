# frankentorch-kgs4.145 - BatchNorm2d f32 Lazy-Zero Input Gradient

Date: 2026-06-21
Agent: cod-a / IvoryDeer

## Lever

Represent f32 BatchNorm2d scalar-loss input gradients as the autograd tape's
lazy zero edge instead of allocating/filling the large `dx` buffer in the custom
backward closure. Preserve the existing f32 `dweight`/`dbias` reduction order via
`batch_norm_backward_scalar_f32_affine_grads`.

## RCH Routing Evidence

- `baseline_batch_norm2d_f32_gauntlet.log`: `vmi1264463` canceled after stale
  progress before measurements completed.
- `baseline2_batch_norm2d_f32_gauntlet.log`: `vmi1153651`, ordinary `165.89 ms`,
  explicit scalar `130.64 ms`; remote PyTorch failed with `No module named
  'torch'`.
- `after_batch_norm2d_f32_gauntlet.log`: `vmi1149989`, ordinary `46.262 ms`,
  explicit scalar `47.163 ms`; remote PyTorch failed with `No module named
  'torch'`.

The remote worker delta was too large to count as same-worker keep proof.

## Same-Machine Keep Evidence

- Patched lazy-zero local run (`after_local_batch_norm2d_f32_gauntlet.log`):
  ordinary `87.938 ms`, explicit scalar `81.685 ms`, PyTorch `8.5874 ms`.
- Temporary dx-allocating API baseline
  (`baseline_local_dx_alloc_batch_norm2d_f32_gauntlet.log`): ordinary
  `94.656 ms`, explicit scalar `85.919 ms`, PyTorch `8.6718 ms`.

Internal ratios:

- Ordinary automatic row: `94.656 / 87.938 = 1.076x` faster. Criterion reported
  the temporary dx-allocating baseline `+9.0622%` slower than patched, `p=0.00`.
- Explicit scalar-sum row: `85.919 / 81.685 = 1.052x` faster, but Criterion
  reported `p=0.28`; neutral/no-count.

Ratios vs PyTorch:

- Patched ordinary: `87.938 / 8.5874 = 10.24x` slower.
- Patched explicit scalar: `81.685 / 8.5874 = 9.51x` slower.

## Correctness

- `test_ft_kernel_cpu_batch_norm_f32_scalar_after.log`: passed
  `batch_norm_f32_scalar_backward_matches_unit_dy_bits` on `hz1`.
- `test_ft_api_functional_batch_norm2d_f32_sum_after_clean.log`: passed
  `functional_batch_norm2d_f32_sum` focused tests on `vmi1149989`.
- Post-rebase `rch exec -- cargo test -p ft-conformance --profile release`:
  passed on `vmi1149989` with `ft-conformance` lib `199/0` plus binaries,
  integration tests, smoke tests, and doctests green.

## Compile And Static Gates

- `check_ft_kernel_cpu_lib.log`: `cargo check -p ft-kernel-cpu --lib` passed on
  `hz1`.
- `check_ft_api_lib.log`: `cargo check -p ft-api --lib` passed on
  `vmi1149989`.
- `clippy_ft_kernel_cpu_lib.log`: `cargo clippy -p ft-kernel-cpu --lib -- -D warnings`
  passed on `hz1`.
- `clippy_ft_api_lib.log`: `cargo clippy -p ft-api --lib -- -D warnings` passed
  on `vmi1149989`.
- `rustfmt_touched_files_check.log`: touched-file rustfmt remains blocked by
  broad pre-existing drift in giant files; no formatting rewrite was included.
- `git diff --check`: passed after the final docs edit.
- `ubs_changed_files_interrupted.md`: changed-file UBS over the giant Rust files
  was interrupted after a long silent run with no findings emitted; the
  docs/artifact-only UBS pass exited `0` with no recognizable languages.

## Verdict

Keep the lazy-zero input-gradient representation for the automatic/native
BatchNorm2d f32 scalar-loss row. Record the explicit scalar-sum row as neutral.
The PyTorch gap remains large; next work should remove report/persistent zero
materialization, deforest the forward output, reuse stats/workspaces, or generate
shape-specialized scalar-loss kernels.
