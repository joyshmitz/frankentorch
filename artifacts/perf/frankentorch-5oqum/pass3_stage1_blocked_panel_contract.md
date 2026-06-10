# frankentorch-5oqum Pass 3: stage-1 blocked-panel proof contract

## Profile-backed target

Pass 2 measured the staged two-stage eigvalsh vehicle on RCH worker
`vmi1227854`:

- Live public guard: `eigvalsh_f64_128x128` `[1.2340 ms 1.4278 ms 1.5676 ms]`.
- Current staged vehicle: `eigvalsh_two_stage_f64_128x128_b16` `[5.4670 ms 5.6085 ms 5.7727 ms]`.
- Stage 1: `sym_to_banded_f64_128x128_b16` `[3.1217 ms 3.3211 ms 3.5014 ms]`.
- Stage 2: `banded_to_tridiag_f64_128x128_b16` `[2.2926 ms 2.4614 ms 2.6477 ms]`.

Stage 1 is the larger measured staged residual. Stage 2 already has the
band-packed `O(n^2 b)` bulge-chase lever. The next implementation lever is
therefore a communication-avoiding stage-1 blocked symmetric-to-banded panel,
not public-path wiring and not additional stage-2 tuning.

## Alien primitive family

Canonical source: `/data/projects/alien_cs_graveyard/alien_cs_graveyard.md`
section `9.6 - Communication-Avoiding Algorithms`.

Applied primitive:

- Replace the unblocked reflector-at-a-time dense-to-banded stage with a
  DLATRD/SBR-style blocked panel.
- Keep safe Rust and the existing local `gemm::dgemm`/`dgemm_bt` kernels.
- Accumulate a panel of reflectors, form compact `V` and `W` panels, then apply
  one symmetric rank-2k trailing update through
  `symmetric_rank2k_lower_update_f64`.
- Keep the existing unblocked algorithm as the strict reference/fallback for
  shapes or spectra that do not clear the proof gate.

Pre-score for the next lever: `Impact 5 * Confidence 3 / Effort 5 = 3.0`.

## Target surface

Primary code surface:

- `crates/ft-kernel-cpu/src/lib.rs`
- `symmetric_to_banded_f64`
- `symmetric_rank2k_lower_update_f64`
- `eigvalsh_two_stage_f64`

Benchmark surface:

- `crates/ft-kernel-cpu/benches/linalg_bench.rs`
- `sym_to_banded_f64_128x128_b16`
- `eigvalsh_two_stage_f64_128x128_b16`
- optional `256x256_b32` confirmation if the 128 row wins cleanly

Do not alter `eigh_contiguous_f64` or `eigvalsh_contiguous_f64` in the next
lever. Public-path wiring is a later pass only after the staged vehicle beats
the live guard and the golden SHA is rechecked.

## Behavior obligations

The blocked stage-1 reducer may change floating-point association inside the
staged-only helper. It must not change public eigensolver behavior until a
later explicit wiring pass.

Required invariants:

- Output band is exactly zero outside `abs(row - col) <= b`.
- Output band remains symmetric within `1e-12`.
- Accumulated `q` remains orthogonal with max error `<= 2e-9`.
- Reconstruction `q @ band @ q^T` matches input with max absolute error
  `<= 5e-9 * max(1, ||A||_inf)`.
- `eigvalsh_two_stage_f64` remains sorted by `f64::total_cmp`.
- No RNG is introduced.
- Shape errors remain `KernelError::ShapeMismatch` for short inputs.
- Non-finite panel data must use the existing rank-2k two-GEMM fallback path
  and must not rely on finite-only one-GEMM symmetry.

Tie/ordering policy:

- Public `eigvalsh_contiguous_f64` remains scalar and bit-pinned in this lever.
- Staged `eigvalsh_two_stage_f64` sorts with `f64::total_cmp`, matching current
  staged behavior.
- Repeated or near-repeated spectra remain a fallback case for any later public
  wiring until sign/tie behavior is explicitly proven.

Floating-point ledger:

- The stage-1 blocked panel is tolerance-ledger work, not bit-exact work.
- Existing `symmetric_rank2k_lower_update_finite_matches_two_gemm_bits` remains
  the bit proof for the finite one-GEMM rank-2k subprimitive.
- Existing `banded_to_tridiagonal_band_packed_is_bit_exact` remains the bit
  proof for stage 2.
- The new blocked stage-1 proof must compare against the unblocked stage-1
  reference by reconstruction and eigenspectrum tolerance, not by direct output
  bits.

## Golden guard

Pre-edit scalar eigvalsh fixture:

- Command: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- env FT_EIGVALSH_GOLDEN=1 cargo run -j 1 -p ft-kernel-cpu --example eigh_golden`
- Log: `pass3_eigvalsh_golden_before.log`
- Fixture: `pass3_eigvalsh_golden_before.txt`
- SHA256: `1870e56ea935f9cc895b24d878db52fe341dc2b195c00656faa38b2db97ac458`

After the next runtime lever, rerun the same command and require the same SHA
unless that lever explicitly wires a new public eigvalsh path. If public wiring
is attempted, the SHA must be replaced by a tolerance-ledger public acceptance
artifact and the old scalar path must remain available as fallback.

## Required gates for the next lever

Correctness and proof:

- `rch exec -- cargo test -j 1 -p ft-kernel-cpu symmetric_to_banded_reconstructs_and_is_banded -- --nocapture`
- `rch exec -- cargo test -j 1 -p ft-kernel-cpu eigvalsh_two_stage_matches_live -- --nocapture`
- `rch exec -- cargo test -j 1 -p ft-kernel-cpu banded_to_tridiagonal_band_packed_is_bit_exact -- --nocapture`
- `rch exec -- cargo test -j 1 -p ft-kernel-cpu symmetric_rank2k_lower_update_finite_matches_two_gemm_bits -- --nocapture`
- `rch exec -- env FT_EIGVALSH_GOLDEN=1 cargo run -j 1 -p ft-kernel-cpu --example eigh_golden`

Hygiene:

- `cargo fmt -p ft-kernel-cpu --check`
- `rch exec -- cargo check -j 1 -p ft-kernel-cpu`
- `rch exec -- cargo clippy -j 1 -p ft-kernel-cpu -- -D warnings`
- `ubs crates/ft-kernel-cpu/src/lib.rs`

Performance:

- Baseline already recorded for `sym_to_banded_f64_128x128_b16`:
  `[3.1217 ms 3.3211 ms 3.5014 ms]` on `vmi1227854`.
- Rebench the same row on `vmi1227854` after exactly one lever.
- Rebench `eigvalsh_two_stage_f64_128x128_b16` on `vmi1227854`.
- Keep only if the stage-1 row wins clearly enough for Score `>= 2.0` and the
  end-to-end staged row moves in the same direction.

## Next pass

Pass 4 may edit runtime code, but only for the single stage-1 blocked-panel
lever. It must not wire the public eigvalsh/eigh paths yet.
