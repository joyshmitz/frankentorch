# frankentorch-gxpb2 pass 7: far-row operation tape rejection

## Verdict

REJECTED. No production source diff remains.

## Profile-backed target

- Bead: `frankentorch-gxpb2`
- Hot row: `eigvals_f64_256x256`
- Worker: `vmi1149989`
- Current profile: n=256 still spends `28.32 ms` in values-only eigvals with
  `319` Francis sweeps, `14` one-by-one deflations, `121` two-by-two
  deflations, no fallback, and no exceptional shifts. n=1024 shows `1132`
  sweeps, so the residual is still the scalar Francis QR floor.

## One lever tried

Values-only pending far-row operation tape inside `eig_francis_schur_traced`.

The candidate kept the scalar shift source, selected-`m` search, deflation
tests, and reflector arithmetic. It applied the current near band inline, then
queued far-column row updates and flushed them before those columns could enter
later column-modification dependencies. Full `eig` stayed on the existing path.

This is not the pass 4 const pipeline, pass 5 branch hoist, or pass 6 active
range trim. It changes the row-update schedule/data movement model for
independent far-column work.

## Behavior proof

- `cargo test -j 1 -p ft-kernel-cpu --lib eig_francis_shadow_profile -- --nocapture`
  passed `3/3` on `vmi1149989` while the candidate was present.
- The shadow oracle proved profile stream equality, Schur-form bit equality,
  eigenvalue bit equality, and full-vector equality for the focused fixtures.
- Ordering/tie behavior: shift samples, selected `m`, active windows,
  exceptional-shift cadence, max-total fallback, deflation order, and
  complex-pair slot order were preserved by construction and by the shadow
  proof.
- Floating point: the candidate used the same scalar `p2` expression and the
  same subtract order for every written slot. It did not use GEMM or reassociate
  sums.
- RNG: none.

## Same-worker benchmark

Baseline command:

`RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench eigvals_f64_256x256 -- --warm-up-time 1 --measurement-time 3 --sample-size 10`

Baseline:

`eigvals_f64_256x256 [26.990 ms 28.819 ms 31.335 ms]`

Candidate command used the same worker and same Criterion row.

Candidate:

`eigvals_f64_256x256 [59.821 ms 63.252 ms 69.052 ms]`

Median ratio:

`28.819 / 63.252 = 0.456x`

Score: `0.0`. The candidate is bit-exact but more than 2x slower, so it fails
the keep gate.

## Decision

The source hunk was removed. `crates/ft-kernel-cpu/src/lib.rs` has no retained
diff for this pass.

## Next route

Do not repeat far-row deferred scheduling on the scalar Francis loop. The next
attack should move to a true sweep-count primitive with strict fallback, not a
same-sweep schedule change:

- partial AED window record with Schur-block reordering and explicit undeflated
  shifts; or
- a small-bulge/multishift kernel isolated behind a copied active-window proof
  harness before public dispatch.

The acceptance gate stays unchanged: strict `eigvals_golden` SHA
`24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`, focused
eig/eigvals tests, same-worker Criterion, and Score >= 2.0.
