# frankentorch-gxpb2 pass6: active-window values-only column projection

## Verdict

KEEP.

One lever shipped: in values-only non-symmetric Francis QR, the column-side
Householder update starts at the active unreduced Schur-window top `l` instead of
row `0`. The vector-producing `eig` path still uses `0..=jmax`, and the shadow
audit records/replays the same `col_start` used by production.

## Profile-backed target

- Bead: `frankentorch-gxpb2`
- Hot row: `eigvals_f64_256x256`
- Profile diagnosis: after q-accumulator and const `WANT_VECTORS` keeps, the
  values-only path still spent work updating inactive coupling rows above the
  current active Schur window during each Francis column transform.

## Benchmark

Primary same-worker A/B, `vmi1149989`:

- Baseline: `[23.153 ms 23.760 ms 24.413 ms]`
- Candidate: `[22.172 ms 22.899 ms 23.793 ms]`
- Median speedup: `1.038x`
- Score: `2.20 = Impact 1.038 * Confidence 0.85 / Effort 0.40`

Artifacts:

- `pass6_baseline_eigvals_256.log`
- `pass6_rebench_active_window_eigvals_256.log`

The confirm run landed on `vmi1152480`, so it is retained as routing evidence
only, not as the keep/reject comparator.

## Behavior proof

- Focused proof: `cargo test -j 1 -p ft-kernel-cpu --lib eig -- --nocapture`
  passed `24/24`.
- Strict golden: extracted `eigvals_golden` SHA stayed
  `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.
- Ordering and tie-breaking: the selected `m`, shift cadence, exceptional shift
  cadence, deflation ordering, and complex-pair slot ordering are unchanged.
- Floating-point: all active-window row and column arithmetic remains in the
  same scalar order. The only skipped writes are values-only inactive coupling
  rows above `l`; those rows are outside the active unreduced Schur window and
  are not consumed by later eigenvalue recurrence.
- RNG: none.

## Gates

- `cargo fmt -p ft-kernel-cpu --check`
- `cargo check -j 1 -p ft-kernel-cpu --all-targets`
- `cargo clippy -j 1 -p ft-kernel-cpu --all-targets -- -D warnings`
- `git diff --check`
- `ubs crates/ft-kernel-cpu/src/lib.rs` exited `0`; UBS reported `0` critical
  issues.

## Follow-on

`frankentorch-gxpb2` remains open because the full size-gated AED/multishift
geev dispatch is not implemented by this small keep. The next lever should stop
micro-tuning the same scalar loop and attack either a strict scalar-shift
operation tape for batched independent far updates or a bounded Schur-window/AED
record with explicit shift/deflation proof.
