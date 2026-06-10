# frankentorch-5oqum Pass 2: staged two-stage eigvalsh profile

## Scope

- Skill loop: `/repeatedly-apply-skill` applying `/extreme-software-optimization`.
- Bead: `frankentorch-5oqum`.
- Worker: `vmi1227854` for every Pass 2 Criterion row.
- Commands were crate-scoped RCH runs: `cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- <row> --warm-up-time 1 --measurement-time 2 --sample-size 10`.
- No runtime source was changed in this pass.

## Rows

| Row | Estimate |
| --- | --- |
| `eigvalsh_f64_128x128` | `[1.2340 ms 1.4278 ms 1.5676 ms]` |
| `eigvalsh_two_stage_f64_128x128_b16` | `[5.4670 ms 5.6085 ms 5.7727 ms]` |
| `sym_to_banded_f64_128x128_b16` | `[3.1217 ms 3.3211 ms 3.5014 ms]` |
| `banded_to_tridiag_f64_128x128_b16` | `[2.2926 ms 2.4614 ms 2.6477 ms]` |

Raw logs:

- `pass2_live_vs_two_stage_128_b16.log`
- `pass2_sym_to_banded_128_b16.log`
- `pass2_banded_to_tridiag_128_b16.log`

## Interpretation

The current two-stage vehicle is not ready to replace the live public path:

- Public live `eigvalsh_f64_128x128` median: `1.4278 ms`.
- Current staged `eigvalsh_two_stage_f64_128x128_b16` median: `5.6085 ms`.
- Ratio: staged is `3.93x` slower at this size.

Inside the staged vehicle, stage 1 is the larger measured cost:

- Stage 1 `symmetric_to_banded_f64`: `3.3211 ms` median.
- Stage 2 `banded_to_tridiagonal_f64`: `2.4614 ms` median.
- Stage 1 share of measured split cost: about `57.4%`.

This matches the current bead note: the band-packed stage-2 bulge chase is already landed, and the next real gate is the stage-1 BLAS-3 blocked panel. More stage-2 tuning is not the profile-backed next lever.

## Behavior/isomorphism proof

- Runtime dispatch is unchanged: no call site was switched to `eigvalsh_two_stage_f64`.
- Ordering, tie-breaking, floating-point evaluation, and RNG behavior on public APIs are unchanged because no runtime implementation path changed.
- Golden-output SHA was not regenerated in this pass; the strict eigensolver golden remains the previously recorded scalar-path SHA until a runtime lever is introduced.
- This pass is an evidence-only routing change, so the behavioral artifact is the absence of runtime source changes plus crate-scoped bench harness evidence from Pass 1.

## Score and route

- Change score: evidence-only, not a kept optimization lever.
- Candidate next lever: stage-1 communication-avoiding blocked symmetric-to-banded reduction, DLATRD/SBR-style panel with BLAS-3 trailing updates in safe Rust.
- Pre-score for next lever: `Impact 5 * Confidence 3 / Effort 5 = 3.0`, above the `>= 2.0` gate.
- Pass 3 deliverable: artifact/proof contract for the stage-1 blocked panel before any runtime code change.
