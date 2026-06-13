# frankentorch-zu2wo pass 1 - shadow column tile replay

Date: 2026-06-13T01:38:00Z
Skill: extreme-software-optimization
Target: `[perf][no-gaps] fql10-C6: blocked/tiled shadow Francis sweep grouping from scalar-complete ledger`

## Lever

Tile the private `FrancisShadowAudit::replay_sweep` column-update pass with the same fixed `TILE` width already used by the row-update shadow replay. This changes only the hidden proof lane; public `eig` and `eigvals` dispatch remain on the production scalar Francis path.

## Baseline

Inherited profiler-backed floor from `frankentorch-x9137` on `hz2`:

- `eigvals_f64_256x256`: `[26.396 ms 26.500 ms 26.737 ms]`
- `eig_f64_256x256`: `[53.720 ms 55.281 ms 57.322 ms]`
- profile: `n=256 sweeps=319 defl1=14 defl2=121 fallback=0 exceptional=0`; `n=1024 sweeps=1132 defl1=18 defl2=503 fallback=0 exceptional=0`

Immediate predecessor smoke from `frankentorch-7ew8d` on `hz2`:

- `eigvals_f64_256x256`: `[26.500 ms 27.002 ms 27.704 ms]`
- `eig_f64_256x256`: `[56.902 ms 57.939 ms 59.784 ms]`

## Re-benchmark

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz2 rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- '^(eigvals_f64_256x256|eig_f64_256x256)$' --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Result on `hz2`:

- `eig_f64_256x256`: `[55.097 ms 56.410 ms 57.408 ms]`
- `eigvals_f64_256x256`: `[26.733 ms 26.978 ms 27.363 ms]`

This is a public-dispatch no-regression smoke for a private proof-lane lever, not a production speedup claim.

## Behavior Proof

- Focused `eig_francis_shadow_profile` tests passed: `3 passed; 0 failed; 446 filtered out`.
- Broader `eig` tests passed: `24 passed; 0 failed; 425 filtered out`.
- Strict `eigvals_golden` stdout SHA-256 stayed `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.
- Ordering: unchanged. Each column block still visits `i` in strictly increasing order, and block boundaries partition the original `0..=step.jmax` traversal without reordering.
- Tie-breaking: unchanged. Shift selection, selected-`m`, deflation paths, complex-pair slot order, and public sorting remain outside this hunk.
- Floating point: unchanged per element. The same `p2` expression and the same three possible stores execute in the same row order with no reassociation or reduction.
- RNG: none introduced or consumed.

## Gates

- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz2 rch exec -- cargo check -p ft-kernel-cpu --lib --examples --benches`: passed on `hz2`.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz2 rch exec -- cargo clippy -p ft-kernel-cpu --lib --examples --benches -- -D warnings`: passed on `hz2`.
- `cargo fmt -p ft-kernel-cpu --check`: passed locally. `rch` refuses `cargo fmt` as a non-compilation command when `RCH_REQUIRE_REMOTE=1`, recorded in `pass1_fmt_check_after_format.log`.
- `ubs crates/ft-kernel-cpu/src/lib.rs`: exit `0`, `0` critical issues.

## Score

Score: `6.0 = Impact 3.0 * Confidence 4.0 / Effort 2.0`

Verdict: keep. This completes the next private blocked/tiled shadow grouping step and preserves the scalar-complete oracle. The next route is a guarded production dispatch experiment or a deeper blocked row/column grouping only after it can be A/B gated against this bit-exact shadow lane.
