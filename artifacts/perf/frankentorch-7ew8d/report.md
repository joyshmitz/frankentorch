# frankentorch-7ew8d Scalar-Complete Shadow Ledger

Date: 2026-06-13T01:16:00Z
Bead: `frankentorch-7ew8d`
Parent: `frankentorch-x9137`

## Result

Completed the scalar-complete private shadow Francis sweep ledger. The replay now starts from the exact scalar post-cleanup Hessenberg state and records each scalar `h[k, k - 1]` subdiagonal overwrite before tiled row/column replay.

Public `eig_impl`, `eigvals_contiguous_f64`, and `eig_contiguous_f64` dispatch remain unchanged.

## Baseline Inherited From x9137

Worker: `hz2`

```text
eigvals_f64_256x256     [26.396 ms 26.500 ms 26.737 ms]
eig_f64_256x256         [53.720 ms 55.281 ms 57.322 ms]
```

Francis profile:

```text
n=256   sweeps=319  defl1=14  defl2=121  fallback=0  exceptional=0
n=1024  sweeps=1132 defl1=18  defl2=503  fallback=0  exceptional=0
```

## One Lever

- moved shadow sweep capture after scalar spike cleanup
- recorded scalar subdiagonal overwrite slots in `FrancisBulgeStep`
- replayed those overwrites before the tiled row/column ledger
- removed unused private sweep window fields

## Proof

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-kernel-cpu --lib eig_francis_shadow_profile -- --nocapture
```

Worker: `hz1`; result: `3 passed; 0 failed`.

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-kernel-cpu --lib eig -- --nocapture
```

Worker: `hz1`; result: `24 passed; 0 failed`.

Strict `eigvals_golden` stdout SHA-256:

```text
24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725
```

Quality gates:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-kernel-cpu --lib --examples --benches
cargo clippy -p ft-kernel-cpu --lib --examples --benches -- -D warnings
cargo fmt -p ft-kernel-cpu --check
ubs crates/ft-kernel-cpu/src/lib.rs
```

Remote clippy was attempted on `hz1` but blocked because `cargo-clippy` is not installed on that worker; local crate-scoped clippy passed. UBS exited 0 with existing broad warnings in `src/lib.rs`.

## Benchmark Smoke

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eigvals_f64_256x256 --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Worker: `vmi1227854`

```text
eigvals_f64_256x256     [25.548 ms 25.781 ms 25.980 ms]
```

This is not a same-worker keep/reject proof against the `hz2` baseline. It is hot-path smoke evidence for a private trace-only change.

## Score

`Impact 4.0 x Confidence 3.0 / Effort 3.0 = 4.0`

The value is proof enablement: scalar-complete replay is now bit-exact and unblocks the next blocked/tiled grouping primitive. No production speedup is claimed in this bead.

## Next Route

File and work the next fql10-C6 bead: use this scalar-complete replay as the oracle for a blocked/tiled grouping in the private shadow lane, then require strict equality and immediate same-worker A/B before public dispatch.
