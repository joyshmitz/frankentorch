# frankentorch-npxbw Pass 2 Multishift Contract

Date: 2026-06-11
Agent: IvoryDeer
Scope: primitive selection and proof contract only. No production source edits in this pass.

## Decision

Selected pass-3 source lever: a private deterministic shift-packet and sweep-trace scaffold for `eig_francis_schur`.

The scaffold should extract the current double-shift source from the live Francis loop into a private/test-only contract surface that records:

- active window `(l, en)` and width,
- ordinary double-shift pair data derived from the trailing 2x2 block,
- exceptional-shift events at the existing `its == 10 || its == 20` cadence,
- start index `m` chosen by the current small-subdiagonal search,
- sweep count, deflation count, fallback/non-convergence count, and `max_total` exhaustion,
- whether the recorded sequence exactly follows the current scalar double-shift arithmetic.

Pass 3 must not implement the public multibulge chase yet. The point is to pin the deterministic shift source and convergence ledger needed by a later two-bulge/four-shift direct small-bulge QR primitive.

## Hotspot Evidence

Pass 1 final same-worker remote-only baseline on `vmi1227854`:

| Row | Criterion interval | Median proxy |
| --- | ---: | ---: |
| `eigvals_f64_256x256` | `[24.799 ms 25.258 ms 25.738 ms]` | `25.258 ms` |
| `eig_f64_256x256` | `[42.861 ms 43.339 ms 43.825 ms]` | `43.339 ms` |

Timing probe on the same worker:

| n | `eigvals` | `eig` | `eig - eigvals` |
| ---: | ---: | ---: | ---: |
| 256 | `28.81 ms` | `46.06 ms` | `17.25 ms` |
| 1024 | `2828.59 ms` | `4784.97 ms` | `1956.38 ms` |

Interpretation: the shared values path is the floor. Full `eig` has extra eigenvector/backsub work, but any sweep-count or far-update batching win in the Francis QR stage should move both rows. This rules out q_acc/backsub-only work for `npxbw`.

Current source boundary:

- `eig_impl` copies the dense square matrix, reduces to upper Hessenberg form, and calls `eig_francis_schur` before optional eigenvector backsubstitution.
- `eig_francis_schur` is an EISPACK-style implicit double-shift Francis QR loop with scalar row/column updates, bottom-up deflation, exceptional shifts, and deferred `q_acc` replay.
- The scalar row update at each bulge step still walks `j in k..row_end`; the column update walks `i in 0..=min(en, k + 3)`. That is the direct sweep surface for later small-bulge multishift work.

## Mapped Sources

- `alien_cs_graveyard.md` section 9.6, Communication-Avoiding Algorithms: the relevant primitive is moving dense linear algebra work from scalar memory traffic toward blocked/batched kernels, while measuring numerical-kernel throughput, convergence iterations, and precision overhead.
- FrankenSuite high-level methodology sections 0.2, 0.4, 0.8, 0.16, and 0.20: profile first, one lever, proof/perf/evidence contracts, statistically comparable benchmarks, proof-carrying artifacts, and conservative fallback.
- `alien-artifact-coding` Family 34, Numerical Linear Algebra: require matrix characterization, method selection, convergence trace or factorization accuracy verification, conditioning/stability notes, and cross-checks against an independent oracle.
- `crates/ft-kernel-cpu/src/lib.rs:12638`: `eig_impl`.
- `crates/ft-kernel-cpu/src/lib.rs:12814`: `eig_francis_schur`.

## Recommendation Contract

Change:
Private deterministic `FrancisShiftPacket` / `FrancisSweepTrace` style scaffold around the existing shift-source logic. It should be callable only from tests/examples/diagnostic code or an internal proof hook. Production `eig_contiguous_f64` and `eigvals_contiguous_f64` must continue through the current legacy double-shift chase.

Hotspot evidence:
The pass-1 same-worker rows above show the shared `eigvals` path is already 58 percent of full `eig` at n=256, and the n=1024 probe shows the same shared QR floor grows into seconds. The target is the shared sweep path, not eigenvector-only replay.

Mapped graveyard sections:
Communication-Avoiding Algorithms section 9.6 for blocked/batched dense linear algebra and explicit convergence/kernel-throughput proof; Family 34 for QR/eigendecomposition proof obligations; FrankenSuite contract sections for evidence-ledger and fallback discipline.

EV score:
Alien EV = `(Impact 3 * Confidence 5 * Reuse 4) / (Effort 2 * AdoptionFriction 1) = 30.0`.

Pass-3 source score:
`Impact 3 * Confidence 5 / Effort 2 = 7.50`.

Priority tier:
A. It is the highest-confidence bridge to direct small-bulge multishift QR, but it is not itself the final performance primitive.

Adoption wedge:
Private/test-only scaffold first. Public dispatch remains legacy double-shift until a later multibulge source lever clears strict proof and same-worker performance gates.

Budgeted mode:
Trace length is bounded by existing `max_total = 60 * n + 100`; any diagnostic allocation must be proportional to the number of recorded sweeps and disabled from production dispatch. On exhaustion, report the current legacy fallback/non-convergence outcome and reject the scaffold if it changes public output.

Expected-loss model:
States are `parity_scaffold`, `hidden_semantic_drift`, `future_multibulge_ready`, and `aed_relapse`. Actions are `keep_legacy_only`, `add_private_shift_scaffold`, `implement_multibulge_now`, and `retry_threshold_aed`. Loss is dominated by semantic drift or changed eigenvalue ordering. The chosen action minimizes expected loss because it creates proof artifacts without changing public arithmetic.

Calibration and fallback trigger:
Fallback to legacy double-shift immediately if the scaffold changes the strict golden SHA, changes iteration/deflation counts for the golden fixture, changes complex-pair slot ordering, requires public dispatch changes, or cannot represent exceptional shifts without arithmetic drift.

Isomorphism proof plan:
Preserve bottom-up `en` deflation order, current `eps * s` split tests, exceptional-shift cadence, `max_total`, conjugate-pair convention, and eigenvalue slot assignment. Pass 3 must prove the trace is observational only by regenerating the strict golden output and comparing the existing SHA-256.

p50/p95/p99 before/after target:
Current pass only selects a scaffold, so no runtime after target is accepted. For the later source lever, beat `eigvals_f64_256x256` median proxy `25.258 ms` and `eig_f64_256x256` median proxy `43.339 ms` on the same worker, with non-overlapping or clearly favorable Criterion intervals and Score `>= 2.0`. Local fallback may route but cannot keep.

Primary failure risk and countermeasure:
Risk is numerical and ordering drift from making the shift source reusable. Countermeasure is test-only isolation plus a strict sequence-parity ledger against the current scalar loop before any multibulge chase or BLAS-3 far update is attempted.

Repro artifact pack:
Use existing pass-1 baseline and golden artifacts under `artifacts/perf/frankentorch-npxbw/`. Pass 3 should add a trace artifact with environment, worker, command, current HEAD, strict golden SHA, and focused test log. No source lever should be accepted without the artifact pack.

Primary paper status:
Hypothesis-level local mapping from the graveyard Demmel/Ballard communication-avoiding summary and standard small-bulge multishift QR lineage. No claim of lower-bound optimality or LAPACK parity is made until a source lever is reproduced against local goldens.

Interference test status:
Not required for the pass-3 scaffold because no controller or public algorithm composition is enabled. Required before compact-WY or BLAS-3 far updates compose with a multibulge chase.

Demo linkage:
N/A. This is a private kernel proof scaffold, not a product-facing feature.

Rollback:
Reject and remove any pass-3 source hunk if the strict golden or focused eig/eigvals tests change. The runtime fallback is the current legacy double-shift `eig_francis_schur`.

Baseline comparator:
Current scalar one-bulge implicit double-shift Francis QR in `eig_francis_schur`.

## Proof Obligations For Pass 3

- No public dispatch change to `eig_contiguous_f64` or `eigvals_contiguous_f64`.
- Strict golden SHA remains `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.
- Focused tests include `eigvals_matches_eig`, `eigvals_companion_complex_roots`, and `eig_parallel_schur_vector_update_matches_single_thread_bit_exact`, or their current exact equivalents if test names drift.
- Shift-packet trace preserves ordinary and exceptional shift arithmetic exactly, including accumulated shift `t`.
- Deflation order and real/complex pair slot convention are unchanged: upper slot of a complex pair has positive imaginary part, lower slot has negative imaginary part.
- The scaffold records convergence iterations and fallback count; it must not reduce, increase, or reorder them while in observational mode.
- No mixed precision, no external BLAS/LAPACK, no source change outside the narrow `ft-kernel-cpu` eig/diagnostic surface for pass 3.
- Crate-scoped validation only if pass 3 changes code: focused tests, `rch exec -- cargo check -p ft-kernel-cpu --all-targets`, `rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`, touched-file fmt/format check, and `ubs` on touched files. No full-workspace builds.

## Rejected Families

- qglh3 values-only AED suffix.
- qglh3 whole-window threshold AED with `q_acc`.
- Active-window threshold trims.
- q_acc/backsub-only work.
- Generic full-matrix TSQR/CAQR. The matrix is already Hessenberg; the relevant communication-avoiding mapping is batched bulge/far updates inside the Francis stage.
- Symmetric `eigvalsh/eigh` or `x53r3` work.
- External LAPACK/BLAS/MKL/XLA adoption.
- Compact-WY accumulated far updates before the deterministic shift-source and convergence trace are proven.

## Why This Is Not Another qglh3 AED Variant

The rejected qglh3 attempts were threshold-deflation families: values-only AED suffix and whole-window AED with `q_acc`. They preserved goldens during parts of the run but failed final same-worker gates and did not deliver the shift-list handoff required by direct multishift QR.

This selected lever does not try to deflate earlier, change thresholds, or replay a window Schur transform. It pins the current deterministic double-shift source and sweep ledger so the next implementation can introduce a direct two-bulge/four-shift small-bulge QR primitive with a known fallback contract. If the scaffold cannot prove exact observational parity, pass 3 should ship no source.
