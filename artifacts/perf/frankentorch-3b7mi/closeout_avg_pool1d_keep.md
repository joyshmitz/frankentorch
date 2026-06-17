# frankentorch-3b7mi closeout - fused f64 avg_pool1d grad route

## Decision

KEEP.

The source change landed in `9551e636` and the evidence bundle landed in
`d375734a`. This closeout records the bead verdict only; no additional source
changes were made during closeout.

## Profile-backed target

- Source profile: `artifacts/perf/frankentorch-ftapi-train-reprofile-20260616/baseline_train_hotspots.log`
- Hot row: `conv1d_family/avg_pool1d_grad`, median `282.43 ms` in the source profile
- Local baseline after the `ts1` offline override:
  `artifacts/perf/frankentorch-3b7mi/pass1_local_baseline_avg_pool1d_grad.log`
  reported `[427.33 ms 435.25 ms 442.14 ms]`

## One lever

Add a dedicated f64 avg-pool1d route:

- `ft-kernel-cpu::avg_pool1d_forward_f64`
- `ft-kernel-cpu::avg_pool1d_backward_f64`
- `FrankenTorchSession::functional_avg_pool1d` f64 fast path, including the
  no-grad leaf shortcut

The fallback 2D reshape route remains in place for non-f64 dtypes.

## Isomorphism proof

- Ordering preserved: output planes are still traversed in `[N, C]` plane order
  and each row writes `ox` ascending.
- Tie-breaking unchanged: no ties are selected by average pooling.
- Floating point preserved: the direct forward sums `kx` ascending and divides
  by the same `kernel_size`; backward visits `ox` ascending then `kx` ascending,
  matching the equivalent `avg_pool2d` route with `kh=1`.
- RNG preserved: no RNG is involved.
- Shape/error behavior preserved: 2-D unbatched input still routes through
  unsqueeze/pool/squeeze, invalid rank and size checks remain unchanged.
- Golden proofs:
  - `pass2e_proof_kernel_avg_pool1d_after_clippy_fix.log`
  - `pass2f_proof_functional_avg_pool1d_after_clippy_fix.log`

## Rebench

Final local Criterion rebench:

- Artifact: `artifacts/perf/frankentorch-3b7mi/pass3c_local_rebench_avg_pool1d_grad_after_clippy_fix.log`
- Candidate: `[246.65 ms 253.52 ms 260.65 ms]`
- Baseline median to candidate median: `435.25 ms -> 253.52 ms`
- Ratio: `1.72x`

Score:

- Impact: `1.72`
- Confidence: `0.95` (same local target dir, focused row, proof-clean)
- Effort: `0.60`
- Score: `2.72`

## Gates

Captured artifacts:

- `pass4_check_ft_api_all_targets.log`
- `pass4_check_ft_kernel_cpu_all_targets.log`
- `pass4_clippy_ft_api_lib.log`
- `pass4b_clippy_ft_kernel_cpu_lib_after_fix.log`
- `pass4_fmt_ft_api_kernel_check.log`
- `pass4_git_diff_check_final.log`
- `pass4_ubs_changed_files_timeout60.log`

Known residual: full `cargo fmt --check` output includes pre-existing unrelated
format drift in `ft-api`; the touched source hunks were already committed and the
final source tree is clean.
