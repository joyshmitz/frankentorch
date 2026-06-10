# frankentorch-34hx9 deep pass 1: residual split profile

## Scope

- Skill loop: `/repeatedly-apply-skill` applying `/extreme-software-optimization`
- Bead: `frankentorch-34hx9`
- Source edits: none
- Benchmark crate: `ft-kernel-cpu`
- RCH policy: remote required, crate-scoped, single cargo job

This pass refreshed the public-path baseline after the rejected delayed-top-left
`eigvalsh` wedge. It also re-ran the existing lower-triangle rank-2k primitive
row because that is the proven BLAS-3 building block for a real blocked
tridiagonalization route.

## Current RCH Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-a rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- 'eigh_f64_256x256|eigvalsh_f64_256x256|sym_rank2k_lower_(scalar|gemm)_f64_256x32' --warm-up-time 1 --measurement-time 5 --sample-size 20
```

RCH selected worker `vmi1227854` at `root@109.123.245.77`; the `RCH_WORKER`
hint was advisory. The worker rewrote `CARGO_TARGET_DIR`, so this was a cold
target build followed by the crate-scoped Criterion rows.

Rows:

- `eigh_f64_256x256`: `[11.991 ms 12.628 ms 13.252 ms]`
- `eigvalsh_f64_256x256`: `[7.5447 ms 7.9339 ms 8.2461 ms]`
- `sym_rank2k_lower_scalar_f64_256x32`: `[792.11 us 852.55 us 910.34 us]`
- `sym_rank2k_lower_gemm_f64_256x32`: `[185.44 us 188.19 us 190.48 us]`

Same-worker median ratios:

- Full-vector `eigh` is `1.59x` the values-only `eigvalsh` path.
- GEMM rank-2k is `4.53x` faster than the scalar lower-rank-2k primitive.

Raw log: `artifacts/perf/frankentorch-34hx9/deep_pass1_current_public_and_rank2k.log`.

## What Can Be Split Without Source Instrumentation

The current Criterion bench exposes only public `eigh`/`eigvalsh` rows and the
standalone rank-2k primitive. The private stages are not directly benchmarkable:

- `eigh_tred2_packed_full`
- `eigh_tred2_values_only`
- `eigh_tql2_transposed`
- `eigh_tql2_values_only`

Therefore this pass can prove:

1. public symmetric eigensolver cost is still material;
2. values-only remains substantially cheaper than full-vector `eigh`;
3. the BLAS-3 rank-2k building block remains high-EV on the same worker.

It cannot directly apportion `eigvalsh` time between dense-to-tridiagonal
reduction and tridiagonal QL without adding a benchmark-only probe or temporary
instrumentation. No such probe was added in this pass.

## Alien Primitive Match

Canonical graveyard match: communication-avoiding linear algebra (§9.6) favors
blocking/tree structure to reduce data movement and convert scalar panel work
into BLAS-3 kernels. The project-local numerical artifact already contains a
separate two-stage vehicle:

- `symmetric_to_banded_f64`: dense symmetric -> symmetric banded;
- `banded_to_tridiagonal_f64`: banded -> tridiagonal by bulge chasing;
- `eigvalsh_two_stage_f64`: end-to-end values-only proof vehicle.

Candidate ranking:

| Candidate | Impact | Confidence | Effort | Score | Verdict |
|---|---:|---:|---:|---:|---|
| Compact-WY blocked `dsytrd` panel over live lower-packed reduction | 5 | 3 | 5 | 3.0 | viable but high-risk |
| Two-stage banded path, first with a benchmark/proof hook and then a band-packed stage-2 lever | 4 | 4 | 4 | 4.0 | best next route |
| Tridiagonal secular/D&C replacement for QL | 4 | 2 | 5 | 1.6 | needs split proof first |

## Pass 2 Direction

Pass 2 should not retry scalar EISPACK loop rearrangements or the rejected
delayed-top-left wrapper. The next profile-backed route is the two-stage banded
primitive family because the code already has correctness tests proving
orthogonal-similarity preservation:

- `symmetric_to_banded_reconstructs_and_is_banded`
- `eigvalsh_two_stage_matches_live`
- `banded_to_tridiagonal_preserves_eigenvalues`

The minimal next lever should be a measurement/proof hook for the two-stage
vehicle or exactly one band-packed stage-2 bulge-chase improvement, with the
public `eigvalsh_f64_256x256` row kept as the same-worker acceptance gate.
