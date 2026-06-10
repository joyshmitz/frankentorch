# Pass 10 Residual Split And Recommendation

## State

- Worktree: `/data/projects/frankentorch-5oqum-boldfalcon`
- HEAD: `a708c9e3` (`perf(ft-kernel-cpu): cap staged eig band panel width`)
- Bead: `frankentorch-5oqum`, status `in_progress`, assignee `BoldFalcon`
- Behavior change in pass10: none. This pass only records evidence and a pass11 recommendation.

## Commands And Workers

- `br show frankentorch-5oqum --json`
  - Confirmed status `in_progress`, assignee `BoldFalcon`.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- 'eigvalsh.*f64_256x256' --sample-size 10 --warm-up-time 1 --measurement-time 3 --noplot`
  - Refused before assignment: `no admissible workers: insufficient_slots=1,active_project_exclusion=1`; remote fallback was required, so no local benchmark was used.
- Observed active pass10 RCH Criterion log: `artifacts/perf/frankentorch-5oqum/pass10_current_head_eigvalsh_components_remote.log`
  - Actual worker: `vmi1227854`.
  - Remote command in log: `cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- 'eigvalsh_f64_256x256|eigvalsh_two_stage_f64_256x256_b32|sym_to_banded_f64_256x256_b32|banded_to_tridiag_f64_256x256_b32' --warm-up-time 1 --measurement-time 2 --sample-size 10`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo run -j 1 -p ft-kernel-cpu --release --example banded_stage1_ab`
  - First attempt on `vmi1227854` failed during dependency download (`fnv` timeout).
  - Offline retry on `vmi1227854` failed because the worker lacked `half`.
  - Successful retry with extended cargo network retry/timeout ran on `vmi1227854`; only this run is used as performance evidence.

## Hotspot Evidence

Same-worker Criterion baseline on `vmi1227854` from the active pass10 log:

| Row | Median |
| --- | ---: |
| `eigvalsh_f64_256x256` | 6.4720 ms |
| `eigvalsh_two_stage_f64_256x256_b32` | 11.491 ms |
| `sym_to_banded_f64_256x256_b32` | 43.001 ms |
| `banded_to_tridiag_f64_256x256_b32` | 21.868 ms |

Stage1 harness on `vmi1227854` (`pass10_current_head_stage1_ab_remote_retry2.log`):

| Shape | Blocked values-only | Unblocked + Q | Speedup |
| --- | ---: | ---: | ---: |
| n128 b16 | 505.77 us | 2319.33 us | 4.59x |
| n256 b32 | 3542.65 us | 45325.10 us | 12.79x |
| n512 b32 | 28426.01 us | 616647.83 us | 21.69x |

Interpretation:

- The post-pass9 staged path is still 1.78x slower than public live at n256 b32: `11.491 ms / 6.472 ms`.
- Stage1 is no longer the only plausible blocker. The pass9 same-worker stage1 value was already about 4.119 ms at n256 b32, and the current successful same-worker `vmi1227854` harness gives 3.543 ms. That leaves roughly 7.95 ms of staged residual after stage1 on the n256 path.
- The existing scalar `banded_to_tridiag_f64` row is not a pass11 candidate: at 21.868 ms it is slower than the whole staged path. It can serve as an oracle, not as the implementation route.

## Recommendation Card

Change:

Implement one private values-only symmetric band primitive for pass11: keep the stage1 output in symmetric band form and reduce it to tridiagonal using a true DSBTRD-style band algorithm with contiguous band-window kernels, then feed the existing values-only tridiagonal QL path. This is not pass8's storage-only compacting and not a scalar EISPACK loop shuffle: the lever is the band-aware algorithmic primitive that avoids sending an already banded matrix through dense packed Householder reduction. Do not wire public dispatch until the same-worker gate passes.

Mapped graveyard sections:

- `/data/projects/alien_cs_graveyard/alien_cs_graveyard.md` section 9.6, "Communication-Avoiding Algorithms": use the data-movement framing and BLAS-3/kernel-throughput proof contract for linear algebra kernels.
- `/home/ubuntu/.codex/skills/alien-artifact-coding/references/34-NUMERICAL-LINEAR-ALGEBRA.md`: eigendecomposition is the right family for symmetric spectral analysis; the artifact contract calls for decomposition records, condition/stability checks, and factorization accuracy verification.
- FrankenSuite summary data-plane rule: deterministic kernels need a conservative fallback; the staged path remains private until the evidence beats the live comparator.

EV score:

- Impact 4: the measured staged/live gap is about 5.019 ms at n256, and the residual is after stage1.
- Confidence 3: the algorithm matches LAPACK's band-to-tridiagonal shape, but constants are the risk at n256.
- Reuse 3: reuses stage1 band output, existing QL, and existing live/two-stage golden tests.
- Effort 4: new numerical primitive plus proof harness.
- Adoption friction 2: private path, no public dispatch until gated.
- EV = `(4 * 3 * 3) / (4 * 2) = 4.5`.

Fallback trigger:

Reject the pass11 lever if either condition holds on the same worker:

- `eigvalsh_two_stage_f64_256x256_b32` median is not at least 8 percent faster than same-run `eigvalsh_f64_256x256`.
- The new band route plus existing QL cannot get the post-stage1 residual under 1.8 ms at n256 b32, using the current same-worker stage1 budget of about 3.54 ms.

Isomorphism proof plan:

- Ordering: final eigenvalues must still be sorted with `f64::total_cmp` ascending.
- Tie handling: near-equal eigenvalues must compare against the live `eigvalsh_contiguous_f64` oracle with the existing tolerance envelope; no unstable tie-dependent API ordering is exposed for values-only output.
- Floating point: arithmetic order will differ from dense packed Householder. Require `eigvalsh_two_stage_matches_live` plus an explicit adversarial near-degenerate fixture; keep the current dense values-only path as a strict fallback for non-finite, tiny, or failed-convergence cases.
- RNG: none.
- Golden SHA: public live `FT_EIGVALSH_GOLDEN=1 cargo run -j 1 -p ft-kernel-cpu --example eigh_golden` must remain `1870e56ea935f9cc895b24d878db52fe341dc2b195c00656faa38b2db97ac458` until public dispatch is intentionally changed.

Before/after target:

- Before: current staged n256 b32 median `11.491 ms` versus public live median `6.4720 ms` on `vmi1227854`.
- After: staged n256 b32 median `<= 5.9 ms` on the same worker, with live same-run comparator recorded; stretch target `<= 5.5 ms`.

Primary risk/countermeasure:

- Risk: band-reduction numerical reordering can drift on clustered eigenvalues or spend too much time in scalar bulge chasing.
- Countermeasure: budgeted private route with dense values-only fallback, near-degenerate fixtures, live-oracle tolerance checks, and an immediate reject if the same-worker n256 constant factors miss the target.

Baseline comparator:

The comparator to beat is the same-run public live `eigvalsh_f64_256x256` Criterion median on the selected worker. For the current pass10 evidence that comparator is `6.4720 ms` on `vmi1227854`; pass9's older `7.8459 ms` live row is historical context only.

Rollback:

Keep pass11 as one private helper and one call-site swap inside the staged two-stage path. If proof or performance fails, revert that commit or gate the helper off and leave public `eigvalsh`/`eigh` unchanged at `a708c9e3` behavior.
