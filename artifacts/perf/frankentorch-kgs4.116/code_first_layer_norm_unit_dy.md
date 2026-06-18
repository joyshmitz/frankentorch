# frankentorch-kgs4.116 code-first perf ledger

## Bead

- `frankentorch-kgs4.116`
- Assignee: `cod-a`
- Lever: `ft-kernel-cpu` LayerNorm backward all-ones upstream-gradient fast path
- Benchmark target: `layer_norm/grad_2048x1024`
- Status: code-first, batch-test pending; no speedup claimed yet

## Baseline Evidence

Corrected grad benchmark decontamination on 2026-06-17 removed RNG from `b.iter`.
The current realistic LayerNorm training trace is `layer_norm/grad_2048x1024`
at about 65 ms in the corrected Criterion run.

The workload comes from `tensor_sum(out)` style loss construction, so the
upstream gradient entering LayerNorm backward is exactly all ones. The generic
kernel still loads `dy`, multiplies by `weight`, multiplies `dy * xhat`, and
serially reduces `dbias` over every row and normalized coordinate.

## Alien-Graveyard Mapping

- Cache/vector execution guidance: remove full-array streams before inventing a
  wider algorithm. This lever deletes the `dy` stream from the hot branch.
- Constants-kill-you rule: the old loop spends work on a mathematically known
  constant gradient; the new branch specializes that constant without changing
  the generic branch.
- Narrow drop-in interface: the public LayerNorm API and saved-stat contracts
  are unchanged; only the CPU kernel selects a more specific implementation
  when `dy` is bit-exact one.

## Implemented Change

`layer_norm_backward_f64` and `layer_norm_backward_f32` now detect
`dy.to_bits() == 1.0.to_bits()` for the full gradient buffer.

When true:

- `dxhat` is computed from `weight` and no longer loads per-element `dy`.
- `dweight` accumulates `xhat` with a literal unit multiplier, preserving the
  generic formula's floating-point shape for the unit-dy case.
- `dbias` is built from one scalar row-count accumulation and copied across
  normalized columns, matching the old repeated `+= 1.0` behavior while avoiding
  `batch * norm_size` additions.
- The generic non-unit-dy branch is untouched.

## Correctness Guard

Added inline unit tests:

- `layer_norm_f64_unit_dy_matches_general_reference_bits`
- `layer_norm_f32_unit_dy_matches_general_reference_bits`

Each test constructs the old generic formula explicitly and compares `dx`,
`dweight`, and `dbias` with raw float-bit equality. These tests are intentionally
kept beside the existing GroupNorm and BatchNorm unit-dy parity guards.

## Negative-Evidence Ledger

| Attempt | Evidence | Result | Do Not Retry As |
| --- | --- | --- | --- |
| LayerNorm saved forward stats | `artifacts/optimization/2026-06-05_ft_api_layer_norm_saved_stats_rejected_frankentorch-yn8y.md` | Rejected: weak/ambiguous speedup and proof gap | Forward-stat rematerialization/saved-stats plumbing |
| RMSNorm unit-dy stat staging | `artifacts/perf/frankentorch-t89dc/closeout_rms_norm_unit_dy_reject.md` | Rejected: same-worker median ambiguous, p=0.58 | RMSNorm all-ones branch with stat staging |
| GroupNorm saved stats | `artifacts/perf/frankentorch-2rsa6/closeout_group_norm_saved_stats_rejected.md` | Rejected | Saved-stat normalization branch as the primary lever |
| BatchNorm1d row-major stats | `frankentorch-2i1cq` | Rejected | Row-major stats rearrangement |

This commit is not a saved-stats retry. It is a constant-gradient specialization
for the already measured LayerNorm grad workload.

## Required Follow-Up Gates

- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a cargo check -p ft-kernel-cpu`
- Batch Criterion on a fixed worker against the current baseline:
  `layer_norm/grad_2048x1024`
- Conformance batch for LayerNorm gradient semantics before closing the bead
- Keep only if same-worker Criterion shows a credible win with parity intact
