# Pass 14 Re-profile And Next Route

## State

- Bead: `frankentorch-5oqum`
- Worktree: `/data/projects/frankentorch-5oqum-boldfalcon`
- Commit profiled: `6a22ec4f`
- Worker: `vmi1227854`
- Command: `cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- 'eigvalsh_f64_256x256|eigvalsh_two_stage_f64_256x256_b32|sym_to_banded_f64_256x256_b32|banded_to_tridiag_f64_256x256_b32' --warm-up-time 1 --measurement-time 2 --sample-size 10`

## Results

| Row | Median |
| --- | ---: |
| `eigvalsh_f64_256x256` | 7.2418 ms |
| `eigvalsh_two_stage_f64_256x256_b32` | 12.877 ms |
| `sym_to_banded_f64_256x256_b32` | 45.898 ms |
| `banded_to_tridiag_f64_256x256_b32` | 20.385 ms |

The two component benches are the existing public Q-accumulating stage-1 and old
banded Givens oracle; they are not the current staged private path. They remain
useful as no-go baselines: neither should be wired into public dispatch.

## Diagnosis

The landed full-lower handoff is still a staged-path improvement, not a public
swap. Public live remains faster on the same worker. Combined with the pass12
private split (`stage1_values` 3705.73 us, `packed_tred2` 6950.99 us,
`values_ql` 254.41 us), the remaining profile-backed target is the
stage1/TRED2 algorithm itself. QL/secular is not the bottleneck.

## Next Primitive

Attack a compact-WY/dsytrd-class blocked tridiagonalization primitive:

- build panels of Householder reflectors over the active lower triangle,
- form the compact WY factor deterministically,
- apply the trailing symmetric rank-2k update through existing safe-Rust
  GEMM/rank-2k kernels,
- preserve strict fallback to the current scalar path for non-finite,
  near-degenerate, or proof-hostile cases,
- prove `total_cmp` ordering, tie behavior, no RNG, and golden SHA before any
  public dispatch consideration.

Target acceptance for the next keep: at least `1.8x` on
`eigvalsh_two_stage_f64_256x256_b32` versus this pass14 staged median, then a
same-worker public comparator before wiring dispatch.
