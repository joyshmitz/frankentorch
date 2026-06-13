# frankentorch-x9137 Pass 3 Shadow Profile Harness

Date: 2026-06-13T00:50:00Z
Bead: `frankentorch-x9137`
Lever: private strict shadow-proof harness for the current Francis QR implementation.

## Source Slice

- Added hidden `EigFrancisShadowProfileResult`.
- Added hidden `eig_francis_shadow_profile_f64`.
- Added bit-exact comparators for:
  - interleaved eigenvalue slots
  - final quasi-Schur buffers
  - `EigFrancisShiftSample`
  - selected-`m` stream
  - active-window stream
  - deflation counters
- Added private tests for values-only and full-`eig` shadow comparisons.
- Public `eig_impl`, `eigvals_contiguous_f64`, and `eig_contiguous_f64` dispatch are unchanged.

The harness currently clones the post-Hessenberg state and runs the scalar traced Schur path as both the reference and shadow lane. This is intentionally a proof scaffold: the next pass can replace only the shadow lane with the blocked/tiled active-window ledger while keeping the current scalar lane as the oracle.

## Behavior Proof

All compile/test commands were crate-scoped to `ft-kernel-cpu`.

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-kernel-cpu --lib eig_francis -- --nocapture
```

Worker: `hz2`

Result:

```text
4 passed; 0 failed
```

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-kernel-cpu --lib eig -- --nocapture
```

Worker: `hz2`

Result:

```text
24 passed; 0 failed
```

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo run -q -p ft-kernel-cpu --release --example eigvals_golden
```

Strict stdout SHA-256:

```text
24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725
```

This matches the pass-1 strict golden SHA.

## Quality Gates

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-kernel-cpu --lib --examples --benches
```

Worker: `vmi1227854`; result: pass.

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-kernel-cpu --lib --examples --benches -- -D warnings
```

Worker: `hz2`; result: blocked because that worker lacks `cargo-clippy`.

```text
cargo clippy -p ft-kernel-cpu --lib --examples --benches -- -D warnings
```

Local fallback result: pass.

```text
cargo fmt -p ft-kernel-cpu --check
```

Result: pass.

```text
ubs crates/ft-kernel-cpu/src/lib.rs
```

Result: exit 0. UBS reported existing broad warnings in this large crate, but no failing finding.

## Benchmark Check

Required after-change benchmark was run:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eigvals_f64_256x256 --sample-size 10 --warm-up-time 1 --measurement-time 3
```

RCH selected `vmi1227854`, not the pass-1 `hz2` worker:

```text
eigvals_f64_256x256     time:   [22.801 ms 23.766 ms 24.579 ms]
```

This row is non-decisive because it is cross-worker relative to the pass-1 `hz2` baseline `[26.396 ms 26.500 ms 26.737 ms]`. It is recorded only as a hot-path-neutral smoke check. The next production-speed pass must use an immediate same-worker before/after pair before any keep/reject scoring.

## Score

Proof-infrastructure score: `7.50 = Impact 3.0 x Confidence 5.0 / Effort 2.0`.

This is not a production speedup claim. The score reflects that the scaffold is now in place to compare a blocked/tiled shadow lane against the scalar oracle without changing public dispatch.

## Next Pass

Replace only the shadow lane inside `eig_francis_shadow_profile_f64` with the active-window blocked/tiled row-column ledger. Keep scalar shift source, selected-`m` search, deflation thresholds, complex-pair slot order, RNG absence, and public dispatch unchanged.
