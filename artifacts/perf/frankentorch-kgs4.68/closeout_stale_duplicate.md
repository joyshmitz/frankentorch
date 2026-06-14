# frankentorch-kgs4.68 closeout

## Verdict

`frankentorch-kgs4.68` is a stale duplicate of work that already landed on
`main`.

- Generic blocked triangular solve landed in `488895e8`:
  `perf(ft-kernel-cpu,ft-api): BLAS-3 blocked triangular_solve for large multi-RHS (kgs4.68)`.
- The exact UPPER Cholesky solve target landed in `bf251c3d`:
  `perf(ft-kernel-cpu): BLAS-3 blocked cholesky_solve UPPER variant for SPD inverse (kgs4.70)`.
- Current `HEAD` during this closeout: `5d562099`.

No source code was changed for this closeout.

## Baseline and score from landed source lever

The committed `bf251c3d` proof recorded same-process A/B against the previous
upper-factor path:

- `n=256`: `3.95 ms -> 2.17 ms`, `1.82x`.
- `n=512`: `29.77 ms -> 10.05 ms`, `2.96x`.
- `n=1024`: `598.0 ms -> 60.31 ms`, `9.92x`.

That source lever cleared the Score `>= 2.0` gate when it landed as
`frankentorch-kgs4.70`.

## Fresh proof commands

Current closeout proof used crate-scoped RCH tests only:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -j 1 -p ft-kernel-cpu --lib cholesky_solve_blocked_path_upper_inverse_n256 -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -j 1 -p ft-kernel-cpu --lib triangular_solve_blocked_matches_serial_n256 -- --nocapture
```

Results:

- `cholesky_solve_blocked_path_upper_inverse_n256`: passed `1/1` on
  `vmi1149989`.
- `triangular_solve_blocked_matches_serial_n256`: passed `1/1` on
  `vmi1227854`.

Log SHA-256s:

```text
1fa249d7a1f41a3c5b9f5e05b1f7f0ad4bf69a2eb07e78cbb66e919e6e511653  test_upper_cholesky_solve_blocked.log
95243a3f13a90a2e16de4dddf10fe60b3dffc98d78cc6fa38e00b9d1b63064a7  test_triangular_solve_blocked_serial.log
```

## Isomorphism proof

- Ordering preserved: yes. The landed blocked solver still solves panels in
  triangular dependency order; only trailing multi-RHS updates are batched through
  GEMM.
- Tie-breaking unchanged: N/A for dense numeric solve.
- Floating-point policy: reassociation is intentional and matches the existing
  dense solve/inverse tolerance policy. The upper Cholesky inverse reconstruction
  test verifies `A @ A^-1` against identity; the triangular-solve proof verifies
  lower and upper blocked results against the serial substitution to working
  precision.
- RNG seeds: N/A; no random inputs or randomized algorithm.
- Golden outputs: the current proof logs are hash-pinned in
  `proof_logs.sha256`; the underlying behavior proof is the deterministic focused
  reconstruction/serial-reference test pair above.

## Close reason

Close `frankentorch-kgs4.68` as completed/superseded by landed commits
`488895e8` and `bf251c3d`. The duplicate open row should not drive a second
source lever.
