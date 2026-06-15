# Pass 4 Rejection: Band-Packed Stage-2 Routing

## Lever Considered

Route `eigvalsh_two_stage_f64` through the existing
`banded_to_tridiagonal_f64` stage-2 bulge chase instead of running
`eigh_tred2_values_only_full_lower` over the band matrix.

No source hunk was applied. The fresh baseline showed the proposed stage-2
primitive is already slower than the whole current two-stage helper.

## Baseline

Same-worker RCH baseline on `vmi1227854`:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- \
  cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- \
  'eigvalsh_f64_256x256|eigvalsh_two_stage_f64_256x256_b32|sym_to_banded_f64_256x256_b32|banded_to_tridiag_f64_256x256_b32' \
  --warm-up-time 1 --measurement-time 3 --sample-size 10
```

| Row | p50 |
| --- | ---: |
| `eigvalsh_f64_256x256` | `7.5550 ms` |
| `eigvalsh_two_stage_f64_256x256_b32` | `12.346 ms` |
| `sym_to_banded_f64_256x256_b32` | `42.391 ms` |
| `banded_to_tridiag_f64_256x256_b32` | `21.047 ms` |

Artifact:

- `artifacts/perf/frankentorch-x53r3/baseline_two_stage_rows_vmi1227854.log`
- SHA256: `8d86403c27fa86816f89571f91a1030421e1c38b3f68c77832b32e825fd06f85`

## Decision

Reject without source edit. Replacing a sub-step with a primitive whose isolated
median (`21.047 ms`) exceeds the current whole helper median (`12.346 ms`) would
be a guaranteed regression and fails the Score gate.

This closes the current `x53r3` dsytrd/WY storage family after prior rejected
passes:

- full-row-major lower `tred2` public routing;
- full-`n^2` compact-WY blocked dsytrd;
- packed-storage wrapper around the same blocked WY panel;
- band-packed stage-2 routing into the existing two-stage helper.

Next route is not another storage wrapper around this reduction. Follow-up
`frankentorch-ggl3d` attacks a fundamentally different primitive:
tridiagonal divide-and-conquer / secular merge with deterministic deflation and
full `eigh_golden` SHA proof.
