# Pass 2 Closeout And Reroute

## Bead

- `frankentorch-92yny`
- Target: `eigh_f64_256x256` eigenvector-side work after `eigvalsh` removes vectors.
- Worker: `vmi1227854`

## Fresh Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -v -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- '^(eigh_f64_256x256|eigvalsh_f64_256x256)$' --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Artifact:

- `artifacts/perf/frankentorch-92yny/pass1_remote_baseline_eigh_eigvalsh_256.log`
- SHA256: `fce4a3deb8873c2be3c9000a25061b45566d30a9e3bdac1408b7994c1b766593`

A duplicate baseline retry was refused while the first baseline was still
active under RCH's project-exclusion policy:

- `artifacts/perf/frankentorch-92yny/pass2_remote_baseline_eigh_eigvalsh_256.log`
- SHA256: `929f20e1bac1e251cc5b714bd1bb03f5e379a90ce4a5a27037225402779e9f85`
- refusal: `critical_pressure=1,active_project_exclusion=1`

Results:

| Row | p50 |
| --- | ---: |
| `eigh_f64_256x256` | `11.066 ms` |
| `eigvalsh_f64_256x256` | `6.5739 ms` |

The isolated with-vectors side is about `4.4921 ms`. Even a perfect zero-cost
eigenvector replay would cap the end-to-end speedup at `11.066 / 6.5739 =
1.68x`, below the campaign's required Score>=2.0 keep threshold.

## Behavior Proof

No source lever was applied. Runtime behavior is unchanged by this closeout.

Existing symmetric eigensolver golden anchor remains the public scalar-path SHA:

- `1870e56ea935f9cc895b24d878db52fe341dc2b195c00656faa38b2db97ac458`
- Prior artifact: `artifacts/perf/frankentorch-5oqum/pass13_full_lower_eigh_golden.sha256`

Fresh remote golden rerun was attempted after the baseline:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -v -- env FT_EIGVALSH_GOLDEN=1 cargo run -j 1 -p ft-kernel-cpu --example eigh_golden
```

RCH refused local fallback because no admissible remote worker was available:
`critical_pressure=1,insufficient_slots=1`.

Artifact:

- `artifacts/perf/frankentorch-92yny/pass2_eigh_golden_remote.log`
- SHA256: `fd6eb7a5a258f23211c8e8bf2668f83fbeadb5a0df6445fd8f08eb787725186a`

## Decision

Reject the isolated `eigh` eigenvector-replay route for this bead. It is not a
runtime ceiling claim: the profile says the wrong target was selected. The next
primitive must attack the shared reduction wall, not the Amdahl tail.

## Next Primitive

Open and attack a band-packed compact-WY/dsytrd primitive:

- panelize Householder formation over packed lower or narrow band storage,
- keep the scalar packed `tred2` fallback for small, non-finite, and
  proof-hostile inputs,
- apply trailing updates with safe-Rust BLAS-3 kernels without expanding to a
  full `n x n` WY footprint,
- preserve ascending `total_cmp` ordering, tie behavior, no RNG, and strict
  `eigh_golden` SHA before public dispatch.

Target ratio: `>=2.0x` on the reduction wall before any public `eigh`/`eigvalsh`
dispatch wiring.
