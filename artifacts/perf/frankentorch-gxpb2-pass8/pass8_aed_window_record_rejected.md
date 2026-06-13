# frankentorch-gxpb2 pass 8: AED window-record substrate rejected

## Target

- Bead: `frankentorch-gxpb2`
- Lane: fql10-D size-gated geev dispatch, preparing AED/multishift sweep-count reduction.
- Lever attempted: add a hidden `eig_francis_aed_window_record_f64` diagnostic substrate that copies the trailing active window, computes a local Schur form/vector basis, records residual/orthogonality/spike metadata, and proves the operation leaves public `eigvals` unchanged.

## Baseline

- Command: `rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench eigvals_f64_256x256 -- --warm-up-time 1 --measurement-time 3 --sample-size 10`
- Log: `artifacts/perf/frankentorch-gxpb2-pass8/pass8_baseline_eigvals_256_vmi1149989.log`
- RCH note: the file name retained the requested worker, but the retrieved benchmark artifact footer is from `vmi1152480`.
- Result: `eigvals_f64_256x256 [25.895 ms 26.068 ms 26.275 ms]`

## Candidate Rebench

- Command: `rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench eigvals_f64_256x256 -- --warm-up-time 1 --measurement-time 3 --sample-size 10`
- Log: `artifacts/perf/frankentorch-gxpb2-pass8/pass8_rebench_eigvals_256_vmi1152480.log`
- Worker: `vmi1152480`
- Result: `eigvals_f64_256x256 [42.165 ms 43.907 ms 45.901 ms]`
- Median ratio: `26.068 / 43.907 = 0.594x`

## Behavior Proof While Candidate Was Present

- Initial focused AED record proof failed only because the strict Schur residual guard was too tight for the copied 32x32 window: `pass8_test_aed_window_record_vmi1152480.log`.
- Corrected focused proof passed: `pass8_test_aed_window_record_retry_vmi1152480.log`, `1 passed; 0 failed`.
- Broad `eig` filter passed: `pass8_test_eig_filter_vmi1152480.log`, `25 passed; 0 failed`.
- Strict golden stdout was unchanged:
  - `pass8_eigvals_golden.strict.stdout`
  - SHA-256 `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`
- The candidate was diagnostic-only, so public ordering, tie-breaking, complex-pair slot layout, shift selection, and RNG absence were intended to stay unchanged. The strict golden and filtered tests confirm the observed public behavior while the hunk was present.

## Verdict

- Rejected. The hidden AED window-record substrate regressed the same public benchmark from `26.068 ms` to `43.907 ms` median on the comparable `vmi1152480` artifact path.
- Score: `0.0`; Impact is negative for the public dispatch.
- Source hunk removed. `crates/ft-kernel-cpu/src/lib.rs` has no retained pass 8 diff after cleanup.

## Reroute

- Do not keep hidden public-code diagnostics that perturb production layout or monomorphization.
- Next pass should either isolate AED proof in a non-linked proof harness, or move directly to a measured sweep-count primitive: bounded AED window with Schur-block reordering/undeflated shifts or a small-bulge/multishift operation tape.
- Keep the focus on reducing Francis sweep count; no more branch/range/const micro-levers for this bead.
