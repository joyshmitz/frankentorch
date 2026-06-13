# frankentorch-gxpb2 pass 8 rejection

Date: 2026-06-13
Agent: IvoryDeer
Bead: frankentorch-gxpb2

## Target

`eigvals_f64_256x256` remains the profile-backed public hotspot for the
fql10-D geev/AED/multishift lane. The pass-7 far-row operation tape proved
bit-exact but was more than 2x slower, so pass 8 tried two deeper directions:

1. a narrow f64x4 row-update candidate inside the values-only Francis sweep;
2. a copied trailing AED/Schur-window proof prototype for the next
   sweep-count-reduction primitive.

## Baseline

RCH Criterion artifact:
`artifacts/perf/frankentorch-gxpb2-pass8/pass8_baseline_eigvals_256_vmi1149989.log`

The log selects `vmi1227854` at the start, though the final RCH footer prints
`vmi1152480`; use the selected-worker line for the intended same-worker
comparison and keep the footer caveat attached to this pass.

```text
eigvals_f64_256x256     time:   [25.895 ms 26.068 ms 26.275 ms]
```

## Candidate A: f64x4 row update

One source lever tried: pack the values-only Francis row update across four
contiguous row slots while preserving the same per-lane scalar expression and
tail order.

Proof while present:

- `cargo test -j 1 -p ft-kernel-cpu --lib eig_francis_shadow_profile -- --nocapture`
  passed `3/3` on `vmi1227854`.
- Strict extracted `eigvals_golden` SHA stayed
  `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.
- Ordering, selected-m/shift/deflation streams, complex-pair slot order,
  floating-point lane expressions, and RNG absence were unchanged.

Same-worker candidate artifact:
`artifacts/perf/frankentorch-gxpb2-pass8/pass8_candidate_simd_row_eigvals_256_vmi1227854.log`

```text
eigvals_f64_256x256     time:   [27.244 ms 27.656 ms 28.100 ms]
```

Median ratio: `26.068 / 27.656 = 0.942x`.

Score: `0.0`; source hunk removed.

## Candidate B: copied AED window record

One proof prototype tried: copy a bounded trailing active window, run the
existing strict Francis Schur routine locally, and record Schur form/vectors,
undeflated shifts, similarity residual, orthogonality residual, and conservative
spike magnitude.

Focused proof artifacts:

- `pass8_aed_window_record_test_vmi1227854.log`: `1/1` passed.
- `pass8_aed_candidate_eig_tests_vmi1227854.log`: `25/25` eig-filter tests passed.
- `pass8_aed_window_record_test_final_vmi1227854.log`: `1/1` passed after the
  local clippy iterator fix.

The later public-path rebench showed the hidden diagnostic hunk still perturbed
the production binary enough to fail the keep gate:

- baseline artifact: `pass8_baseline_eigvals_256_vmi1149989.log`
- candidate artifact: `pass8_rebench_eigvals_256_vmi1152480.log`
- result: `[25.895 ms 26.068 ms 26.275 ms]` -> `[42.165 ms 43.907 ms 45.901 ms]`
- median ratio: `0.594x`

This candidate therefore fails as both proof infrastructure and production
surface. The final closeout has no retained `crates/ft-kernel-cpu/src/lib.rs`
diff from pass 8.

## Verdict

Rejected. No source code retained.

Next route: stop pursuing row-packing/micro-loop candidates. Attack the
sweep-count primitive directly: a strict copied-window AED lane with explicit
deflatable suffix/reordered Schur-block proof and production dispatch only
after the strict golden SHA and same-worker Criterion Score gate clear.
