# frankentorch-ggl3d: vector-only tridiagonal D&C rejected

Scope: `ft-kernel-cpu` full-vector symmetric eigensolver, target row
`eigh_f64_256x256`.

This bead proposed replacing the post-tridiagonal full-vector QL/eigenvector
rotation path with a tridiagonal divide-and-conquer / secular merge. The fresh
same-worker split shows that target is now capped below the public keep gate.

## Baseline

Worker: `ovh-a`

Commands:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- eigh_f64_256x256 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-a rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- eigvalsh_f64_256x256 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Rows:

| Row | Median |
| --- | ---: |
| `eigh_f64_256x256` | `9.7874 ms` |
| `eigvalsh_f64_256x256` | `5.6248 ms` |

Raw logs:

- `artifacts/perf/frankentorch-ggl3d/baseline_eigh_f64_256.log`
- `artifacts/perf/frankentorch-ggl3d/baseline_eigvalsh_f64_256.log`

## Decision

The full-vector-only residual is `9.7874 - 5.6248 = 4.1626 ms`. Even a perfect
replacement for every vector-only QL/merge cost would cap the public-row speedup
at `9.7874 / 5.6248 = 1.74x`, below the required `2.0x` keep threshold before
implementation risk.

This is rejected as a standalone public dispatch lever. It is not a ceiling:
the next admissible symmetric-eigensolver primitive must attack the shared
reduction/values path as well, for example a band-packed SBR / compact-WY
tridiagonalization route that reduces the `eigvalsh` baseline before a D&C merge
is wired into full-vector `eigh`.

## Behavior Proof Status

No runtime source changed. Ordering, tie-breaking, floating-point evaluation, RNG
use, and golden outputs remain exactly at the current committed implementation.

Score: `Impact 3 x Confidence 5 / Effort 5 = 3.0` for routing value only; no
source keep.
