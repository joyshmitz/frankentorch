# frankentorch-7ew8d Pass 1 Scalar-Complete Shadow Replay

Date: 2026-06-13T01:16:00Z
Bead: `frankentorch-7ew8d`
Lever: private scalar-complete shadow replay ledger for Francis QR sweeps.

## Result

Kept one hidden proof-harness lever in `crates/ft-kernel-cpu/src/lib.rs`.

The previous blocked/tiled shadow replay failed because the replay snapshot and step ledger missed scalar state writes made by the live sweep. This lever makes the shadow replay scalar-complete for the current sweep order:

- snapshot the Hessenberg buffer after the sub-sub-diagonal cleanup loop;
- record the exact subdiagonal assignment made before each bulge row/column transform;
- replay that assignment before applying the existing ordered row and column updates.

Public `eig_impl`, `eigvals_contiguous_f64`, and `eig_contiguous_f64` dispatch remain unchanged.

## Baseline

Inherited x9137 baseline on same RCH worker `hz2`:

- `eigvals_f64_256x256`: `[26.396 ms 26.500 ms 26.737 ms]`
- `eig_f64_256x256`: `[53.720 ms 55.281 ms 57.322 ms]`
- strict golden SHA-256: `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`

## Proof

All validation was crate-scoped.

- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz2 rch exec -- cargo test -p ft-kernel-cpu --lib eig_francis_shadow_profile -- --nocapture`
  - Worker: `hz2`
  - Result: `3 passed`
  - Shadow replay mismatches: `0`
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz2 rch exec -- cargo test -p ft-kernel-cpu --lib eig -- --nocapture`
  - Worker: `hz2`
  - Result: `24 passed`
- strict `eigvals_golden` extracted stdout SHA-256:
  - `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`

Isomorphism:

- Ordering preserved: yes; scalar sweep order and selected-`m` stream are unchanged.
- Tie-breaking unchanged: yes; eigenvalue slot ordering and complex-pair ordering are unchanged.
- Floating-point: public eig/eigvals path is unchanged; hidden replay uses the same ordered assignments and row/column transforms as the scalar path.
- RNG: none.
- Golden outputs: strict stdout SHA stayed `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.

## Production Benchmark Smoke

Same-worker smoke rebench on `hz2`:

- `eig_f64_256x256`: `[56.902 ms 57.939 ms 59.784 ms]`
- `eigvals_f64_256x256`: `[26.500 ms 27.002 ms 27.704 ms]`

This is not a production speedup claim. The lever only changes the hidden proof harness path, so the benchmark is a public-dispatch no-regression/routing check. The next bead must attempt the first blocked/tiled grouping and score it against an immediate same-worker before/after pair.

## Quality Gates

- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz2 rch exec -- cargo check -p ft-kernel-cpu --lib --examples --benches`: pass
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz2 rch exec -- cargo clippy -p ft-kernel-cpu --lib --examples --benches -- -D warnings`: pass
- `cargo fmt -p ft-kernel-cpu --check`: pass
- `ubs crates/ft-kernel-cpu/src/lib.rs`: exit `0`; no critical issues, existing broad warning inventory only

## Score

Proof-infrastructure score:

```text
Impact 3.0 * Confidence 5.0 / Effort 2.0 = 7.50
```

This clears the keep gate as prerequisite infrastructure for the blocked/tiled Francis sweep path. It is not a runtime speedup score.

## Next Route

Open a successor bead for the first blocked/tiled shadow grouping. It must preserve the scalar-complete replay identity proven here before any public dispatch wiring, then run a true same-worker before/after benchmark and keep only if Score >= 2.0.
