# frankentorch-92yny pass 1 symmetric eigensolver route

Date: 2026-06-11
Agent: BlackThrush
Bead: frankentorch-92yny

## Scope

This pass claimed the only ready perf bead after fql10 was coordinated and
qglh3/npxbw were blocked by ownership/dependencies. It captured a fresh remote
baseline and decides whether an eigenvector-only `eigh` lever is admissible.

No runtime source changed in this pass.

## Remote baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -v -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- '^(eigh_f64_256x256|eigvalsh_f64_256x256)$' --sample-size 10 --warm-up-time 1 --measurement-time 3
```

RCH selected `vmi1227854`.

| Row | Estimate |
| --- | ---: |
| `eigh_f64_256x256` | `[10.662 ms 11.066 ms 11.559 ms]` |
| `eigvalsh_f64_256x256` | `[6.3973 ms 6.5739 ms 6.7658 ms]` |

Log SHA-256:

```text
fce4a3deb8873c2be3c9000a25061b45566d30a9e3bdac1408b7994c1b766593  artifacts/perf/frankentorch-92yny/pass1_remote_baseline_eigh_eigvalsh_256.log
```

The vector-only delta is about `4.492 ms`. Even deleting all vector-only work
would cap this exact public row at about `1.68x`, and previous ncwz work already
shipped the big vector-layout lever (`eigh_tql2_transposed`).

## Evidence respected

- `frankentorch-ncwz` already kept transposed tql2 rotation layout and moved
  full `eigh` from old `77.522 ms` class to the current `~11 ms` class.
- `frankentorch-t0b4l` rejected full-n^2 compact-WY dsytrd: worker-dependent
  and never above the keep gate because bandwidth and full-matrix footprint beat
  the flop reduction.
- `frankentorch-5oqum` rejected the current two-stage route through n=1024
  because dense-to-band plus band-to-tridiag did too much work.

## Decision

Do not implement an eigenvector-only source hunk for 92yny. It is profile-backed
as a small residual, but it does not attack the shared reduction wall and would
not move `eigvalsh`.

This is not a ceiling. The next admissible symmetric-eigensolver primitive is a
different algorithmic shape:

- band-packed SBR/dsytrd-style reduction that avoids the rejected full-n^2 WY
  footprint and keeps traffic closer to `O(n^2 * b)`;
- tridiagonal divide-and-conquer / secular-equation merge for full-vector `eigh`
  once the reduction/values path is no longer the dominant wall.

## Proof and routing contract

- Ordering: keep ascending `total_cmp` eigenvalue order.
- Ties/signs: any full-vector path must document deterministic vector
  orientation/sign behavior or use reconstruction/orthogonality tolerance per
  qgce4.
- Floating point: band-packed SBR and D&C reassociate, so require tolerance
  parity, golden SHA before/after, and reconstruction/orthogonality tests.
- RNG: none.
- Performance gate: compare both `eigvalsh_f64_256x256` and
  `eigh_f64_256x256` on the same worker; a vector-only row cannot close the
  deeper no-gaps target by itself.

## Result

Close 92yny as a routing/rejection bead and file/attack the deeper band-packed
SBR/D&C work instead of spending a source commit on the capped vector-only
residual.
