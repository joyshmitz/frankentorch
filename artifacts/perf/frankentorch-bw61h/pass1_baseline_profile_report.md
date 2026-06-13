# frankentorch-bw61h pass 1 baseline/profile

Bead: `frankentorch-bw61h`
Worker: `hz1` for timing/profile/proof tests unless noted.

## Criterion baseline

- First unpinned `hz1` route for context:
  - `eigvals_f64_256x256`: `[34.532 ms 35.038 ms 35.659 ms]`
- Pinned artifact rows:
  - `eigvals_f64_256x256`: `[45.990 ms 47.286 ms 48.538 ms]`
  - `eig_f64_256x256`: `[77.037 ms 78.321 ms 79.717 ms]`

The pinned row is the pass-1 artifact baseline, but the keep gate for any source
lever must use an immediate same-worker before/after pair because `hz1` showed
material timing drift across consecutive runs.

## Profile evidence

`eig_timing_probe` on `hz1`:

- `n=128`: eigvals `6.29 ms`, eig `11.55 ms`, sweeps `173`, defl1 `28`, defl2 `50`, fallback `0`, exceptional `0`, max_width `128`
- `n=256`: eigvals `53.41 ms`, eig `75.51 ms`, sweeps `319`, defl1 `14`, defl2 `121`, fallback `0`, exceptional `0`, max_width `256`
- `n=512`: eigvals `458.19 ms`, eig `752.83 ms`, sweeps `583`, defl1 `10`, defl2 `251`, fallback `0`, exceptional `0`, max_width `512`
- `n=1024`: eigvals `3034.61 ms`, eig `5728.78 ms`, sweeps `1132`, defl1 `18`, defl2 `503`, fallback `0`, exceptional `0`, max_width `1024`

Diagnosis: the measured floor is still the scalar Francis QR sweep stream. The
candidate source pass must preserve shift packets, selected `m`, active-window
sequence, deflation counters, quasi-Schur buffer bits, eigenvalue/eigenvector
bits, and complex-pair slot order.

## Golden and proof

- Strict `eigvals_golden` stdout SHA-256:
  `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`
- Printed digests:
  - n=64 eigvals/eig: `0xbc0583d464b1a211`
  - n=128 eigvals/eig: `0x763c4b15d92c4b89`
  - n=256 eigvals/eig: `0x00b87b4996340204`
- `cargo test -p ft-kernel-cpu --lib eig_francis_shadow_profile -- --nocapture`: `3 passed`
- `cargo test -p ft-kernel-cpu --lib eig -- --nocapture`: `24 passed`

## Artifact files

- `pass1_baseline_eigvals_hz1.log`
- `pass1_baseline_eig_hz1.log`
- `pass1_eig_timing_probe_hz1.log`
- `pass1_eigvals_golden_vmi1152480.extracted.stdout`
- `pass1_eigvals_golden_vmi1152480.extracted.stdout.sha256`
- `pass1_shadow_profile_tests_hz1.log`
- `pass1_eig_tests_hz1.log`
