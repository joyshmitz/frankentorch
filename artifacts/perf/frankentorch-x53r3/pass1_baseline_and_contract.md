# Pass 1 Baseline And Contract

## Bead

- `frankentorch-x53r3`
- Target: band-packed compact-WY/dsytrd reduction for `eigvalsh`/`eigh`.
- Source scope: `crates/ft-kernel-cpu/src/lib.rs` only for any first source lever.

## Profile-Backed Target

This bead is the deeper replacement for the closed `frankentorch-92yny` route.
The fresh same-worker baseline shows the shared reduction wall dominates the
with-vectors path:

| Row | p50 |
| --- | ---: |
| `eigh_f64_256x256` | `11.066 ms` |
| `eigvalsh_f64_256x256` | `6.5739 ms` |

Baseline command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -v -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- '^(eigh_f64_256x256|eigvalsh_f64_256x256)$' --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Baseline artifact:

- `artifacts/perf/frankentorch-92yny/pass1_remote_baseline_eigh_eigvalsh_256.log`
- SHA256: `fce4a3deb8873c2be3c9000a25061b45566d30a9e3bdac1408b7994c1b766593`

The isolated eigenvector-side route has a perfect-case cap of `1.68x`, so this
bead attacks the shared `tred2` reduction instead.

## Prior Negative Results To Avoid

- Full `n x n` compact-WY reduction (`eigvalsh_blocked_f64`) is retained only as
  a negative-result harness; it loses cache locality and is memory-bandwidth
  bound.
- Two-stage dense-to-band plus band-to-tridiag was rejected through n=1024:
  it performs two cubic reductions, so the BLAS-3 stage-1 benefit does not pay
  for the extra reduction.
- Per-step rayon fanout inside packed scalar `tred2` regressed and is not a
  viable lever.

## Candidate Lever

Build the first band-packed dsytrd panel primitive without public dispatch:

- store reflector panels in packed lower or narrow trailing-band layout,
- form a deterministic compact-WY panel in safe Rust,
- apply the trailing symmetric update through existing safe-Rust GEMM/rank-2k
  kernels over packed/band slices rather than full `n x n` scratch,
- keep current scalar packed `tred2` as the fallback for small, non-finite,
  near-degenerate, and proof-hostile inputs.

This first lever should be a private helper plus focused proof harness, not a
public `eigvalsh` dispatch swap, unless it already clears the same-worker
Score>=2.0 gate.

## Isomorphism Contract

- Eigenvalues remain sorted by `f64::total_cmp`.
- Equal-value tie behavior remains deterministic after the final sort.
- No RNG or data-dependent nondeterministic scheduling.
- Floating-point arithmetic may reassociate only behind a tolerance-parity gate;
  strict public dispatch keeps the scalar packed path until the golden SHA and
  reconstruction proof pass.
- Golden anchor: public `eigh_golden` SHA
  `1870e56ea935f9cc895b24d878db52fe341dc2b195c00656faa38b2db97ac458`.

## Acceptance Gate

Keep only if a same-worker RCH rebench shows Score>=2.0. The initial target is
`>=2.0x` on the reduction wall before any public dispatch wiring.
