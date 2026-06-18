# frankentorch-5oqum pass 4 active-window keep

## Change

Narrow `symmetric_to_banded_f64` two-sided Householder application to the active trailing window:

- left apply columns `k..n` instead of `0..n`
- right apply rows `k..n` instead of `0..n`

This is a stage-1 two-stage eigensolver vehicle change only. Public
`eigvalsh_contiguous_f64` and `eigh_contiguous_f64` dispatch remain on the
existing scalar path.

## Profile target

Pass 2 same-worker baseline on `vmi1227854`:

- `sym_to_banded_f64_128x128_b16`: `[3.1217 ms 3.3211 ms 3.5014 ms]`
- `eigvalsh_two_stage_f64_128x128_b16`: `[5.4670 ms 5.6085 ms 5.7727 ms]`

Pass 4 same-worker rebench on `vmi1227854`:

- `sym_to_banded_f64_128x128_b16`: `[2.2923 ms 2.3667 ms 2.4216 ms]`
- `eigvalsh_two_stage_f64_128x128_b16`: `[4.4910 ms 4.6295 ms 4.7635 ms]`

Median speedups:

- stage-1 row: `3.3211 / 2.3667 = 1.40x`
- composite staged row: `5.6085 / 4.6295 = 1.21x`

Score: `Impact 3 * Confidence 4 / Effort 2 = 6.0`; keep gate clears `>= 2.0`.

## Isomorphism proof

- Ordering preserved: public eigensolver path is unchanged; staged two-stage path
  still sorts through the existing `eigvalsh_two_stage_f64` flow and passes the
  focused live-comparison test.
- Tie-breaking unchanged: public `total_cmp` ordering/tie behavior is unchanged.
  The staged vehicle is not wired into public dispatch.
- Floating point: the staged band reducer intentionally skips finalized
  out-of-band rows/columns; it is tolerance-equivalent, not bit-identical to the
  previous full-window staged reference. Public scalar golden output remains
  byte-identical.
- RNG: unchanged; no RNG is introduced.
- Golden outputs: pre/post `FT_EIGVALSH_GOLDEN=1` payload SHA-256 is
  `1870e56ea935f9cc895b24d878db52fe341dc2b195c00656faa38b2db97ac458`.

## Verification

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -j 1 -p ft-kernel-cpu symmetric_to_banded -- --nocapture`
  passed on `vmi1227854`.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -j 1 -p ft-kernel-cpu eigvalsh_two_stage_matches_live -- --nocapture`
  passed on `vmi1227854`.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -j 1 -p ft-kernel-cpu bit_exact -- --nocapture`
  passed on `vmi1227854` with 23 tests.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -j 1 -p ft-kernel-cpu`
  passed on `vmi1227854`.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -j 1 -p ft-kernel-cpu -- -D warnings`
  passed on `vmi1227854`.
- `cargo fmt -p ft-kernel-cpu --check` passed.
- `ubs crates/ft-kernel-cpu/src/lib.rs` reported 0 critical issues; broad
  pre-existing warnings remain.
- `git diff --check -- crates/ft-kernel-cpu/src/lib.rs .skill-loop-progress.md .beads/issues.jsonl artifacts/perf/frankentorch-5oqum`
  passed.

## Closeout decision

Keep the active-window stage-1 lever. Do not close `frankentorch-5oqum` yet:
the public `eigvalsh/eigh` path has not moved, and the staged row is still
slower than live. The next no-gaps primitive remains a true DLATRD/SBR-style
blocked dense-to-banded panel with deferred `b` reflectors and a BLAS-3
rank-2b trailing update, followed by a public values-only fast-path gate.
