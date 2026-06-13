# frankentorch-x9137 Pass 2 Shadow Sweep Contract

Date: 2026-06-13T00:36:00Z
Bead: `frankentorch-x9137`

## Measured Failure Signature

Pass 1 on RCH worker `hz2` established:

- `eigvals_f64_256x256`: `[26.396 ms 26.500 ms 26.737 ms]`
- `eig_f64_256x256`: `[53.720 ms 55.281 ms 57.322 ms]`
- strict `eigvals_golden` stdout SHA-256: `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`
- `eig_francis_profile_f64` n=256: `sweeps=319 defl1=14 defl2=121 fallback=0 exceptional=0 max_width=256 truncated=false`
- `eig_francis_profile_f64` n=1024: `sweeps=1132 defl1=18 defl2=503 fallback=0 exceptional=0 max_width=1024 truncated=false`

This is a clean scalar Francis QR sweep floor: no fallback and no exceptional shifts on the target fixture.

## Canonical Source Mapping

Primary graveyard primitive: `/data/projects/alien_cs_graveyard/alien_cs_graveyard.md` section 9.6, Communication-Avoiding Algorithms.

Relevant points:

- minimize data movement by panel/tree/block structure instead of scalar one-column or one-update communication patterns;
- dense submatrix work should be promoted into BLAS-3-like tiled kernels once correctness is proven;
- implementation complexity and constants are the risk, so a shadow/proof harness must precede production dispatch.

Second canonical summary scan:

- `/data/projects/alien_cs_graveyard/high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` did not expose a FrankenTorch-specific QR/eigen route under the searched terms;
- use project-local measured evidence and the canonical §9.6 primitive rather than inventing a broader project mapping.

Alien-artifact family:

- Numerical linear algebra proof obligations apply: factorization/solver accuracy, convergence trace, stability bounds, independent cross-checks, and explicit fallback on ill-conditioned or unproven paths.
- This bead is strict-mode, so the proof target is stronger than tolerance parity: identical stream and golden digest before any production path.

## Candidate Matrix

| Candidate | Impact | Confidence | Effort | Score | Decision |
|---|---:|---:|---:|---:|---|
| Private scalar-shift-preserving shadow active-window ledger harness | 3 | 5 | 2 | 7.50 | Do next |
| Blocked/tiled row-column ledger inside the private harness | 4 | 4 | 3 | 5.33 | Do after harness |
| Public dispatch to blocked sweep kernel | 5 | 2 | 5 | 2.00 | Blocked until strict proof |
| Range/index micro-cuts | 1 | 1 | 1 | 1.00 | Rejected by 8ubk3 regression |
| Alternate shift packets / AED shift-list | 4 | 1 | 4 | 1.00 | Rejected by fy8to digest change |
| Diagnostic-only counters | 1 | 5 | 1 | 5.00 | Rejected as already sufficient |

## Selected Pass-3 Lever

Implement exactly one private proof harness surface:

- clone or duplicate the active Hessenberg state outside public dispatch;
- run the current scalar shift source and selected-m search unchanged;
- emit enough ordered per-sweep/per-bulge update facts to compare a shadow ledger against the scalar path;
- keep `eig_impl`, `eig_contiguous_f64`, and `eigvals_contiguous_f64` public dispatch unchanged;
- expose the harness only through `#[doc(hidden)]` API or `#[cfg(test)]` tests.

Pass 3 may add trace fields or private helper structures if needed, but it must not change production eigenvalue results or the benchmark row.

## Proof Obligations

For every source slice:

- `EigFrancisShiftSample` stream identical by `f64::to_bits`;
- selected-`m` stream identical;
- active-window stream identical;
- deflation counters identical;
- fallback and exceptional-shift counters identical;
- complex pair slot ordering unchanged;
- output eigenvalue interleaved slots identical by bits for the strict fixture;
- RNG absent;
- strict `eigvals_golden` stdout SHA remains `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.

## Budgeted Mode / Fallback

The shadow harness is diagnostic/proof-only. Production behavior remains the current scalar path unless the harness proves strict identity and a later same-worker benchmark clears Score >= 2.0.

Fallback triggers:

- any trace stream mismatch;
- any strict digest mismatch;
- any focused eig/eigvals test failure;
- any RCH local fallback on a decisive proof/bench command;
- any after benchmark below Score >= 2.0.

## Implementation Boundary

Allowed files:

- `crates/ft-kernel-cpu/src/lib.rs`
- optional focused artifacts under `artifacts/perf/frankentorch-x9137/`

Rejected for pass 3:

- changes to public dispatch;
- changes to shift selection;
- changes to deflation thresholds;
- changes to `q_acc` replay;
- broad benchmark additions;
- whole-workspace validation.

## Target Ratio

Pass 3 is proof infrastructure, so the runtime target is neutral. Pass 4 should use the harness to attempt a tiled ledger whose target is at least `1.25x` on `eigvals_f64_256x256` and whose Score must be >= 2.0 before it can remain in source.
