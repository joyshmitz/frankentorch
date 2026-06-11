# frankentorch-ct2yy pass 3 QR stage split

Date: 2026-06-11

## Implementation

One instrumentation lever only:

- Added doc-hidden `QrStageTimings` and `QrStageProfile`.
- Added `qr_contiguous_f64_stage_profile`, which validates inputs like
  `qr_contiguous_f64`, runs the existing compact-WY QR path for the same large
  matrix gate, and returns the profiled Q/R output.
- Threaded optional timers through the existing blocked QR helper. Production
  `qr_contiguous_f64` still calls the same helper with profiling disabled.
- Added `examples/qr_stage_split.rs` to generate the same deterministic
  `512x512` and `2048x128` matrices as `linalg_bench`, run seven samples, and
  print machine-readable stage percentages.

## Isomorphism proof

- Ordering preserved: yes. Production QR dispatch and the unprofiled call path
  are unchanged; timers are only created when the profile helper is called.
- Tie-breaking unchanged: not applicable. QR has no pivoting or tie-breaking
  path changed in this pass.
- Floating-point order: production path unchanged. Profile helper calls the same
  compact-WY math and compares Q/R against production bit-for-bit.
- RNG seeds: not applicable. Matrices are deterministic formulas.
- Golden output: `qr_stage_split` printed `matches_production=true` for both
  profiled shapes. Probe log sha256:
  `c517bceb1e20d13c3515dc3082d46adde5c99da4da269c7fc5f726ab6b57cf09`.

## Stage split

The final probe ran through `rch exec` local fallback because RCH refused remote
assignment (`critical_pressure=1`, `insufficient_slots=1`). Treat these numbers
as routing evidence, not same-worker keep/reject proof.

| Shape | Total | Copy/zero | Panel + T | Trailing R | Reverse Q | Unaccounted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512x512 | 57.326 ms | 1.956% | 45.574% | 17.807% | 32.202% | 2.462% |
| 2048x128 | 28.420 ms | 0.523% | 70.333% | 9.988% | 18.937% | 0.219% |

Both rows were `blocked=true` and `matches_production=true`.

## Pass 4 gate

Local fallback evidence supports the pass-2 diagnosis: panel factorization plus
T build is the dominant stage, especially on the tall row.

Pass 4 is not fully authorized by the pass-2 trigger yet because pass 2 required
same-worker remote split evidence before implementing recursive/block-recursive
panel QR. The next action is to rerun `qr_stage_split` under remote RCH,
preferably on `vmi1227854`; if panel plus T remains at or above 35% on either
primary row, recursive/block-recursive panel QR clears the trigger.

## Gates

- `rustfmt --edition 2024 --check crates/ft-kernel-cpu/src/lib.rs crates/ft-kernel-cpu/examples/qr_stage_split.rs`: pass.
- `RCH_REQUIRE_REMOTE=1 rch exec -v -- cargo check -p ft-kernel-cpu --all-targets`: remote `vmi1227854` ran but failed on pre-existing untracked `crates/ft-kernel-cpu/examples/qr_probe.rs`.
- `rch exec -v -- cargo check -p ft-kernel-cpu --lib --tests --benches --example qr_stage_split`: pass via local fallback.
- `rch exec -v -- cargo test -p ft-kernel-cpu --lib qr_blocked_tall_reconstructs_and_orthonormal`: pass via local fallback.
- `rch exec -v -- cargo clippy -p ft-kernel-cpu --lib --tests --benches --example qr_stage_split -- -D warnings`: pass via local fallback.
- `ubs crates/ft-kernel-cpu/src/lib.rs crates/ft-kernel-cpu/examples/qr_stage_split.rs`: exit 0; no critical findings, broad existing warnings recorded.
