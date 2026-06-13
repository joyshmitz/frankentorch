# rejected lever: packed f64 dgemm_bt B^T panels

## Target

Profile-backed fallback target after no ready `[perf]` bead was available in
the stale scratch tracker:
`matmul_rhs_transposed_contiguous_f64` / Linear-style `dgemm_bt`.

Routing profile on RCH `vmi1227854`, commit `c0b6f3da`:

- `linear x[512,1024] @ W[1024,1024]^T`: `5.589 ms`, `192 GFLOP/s`
- `linear x[1024,1024] @ W[1024,1024]^T`: `11.276 ms`, `190 GFLOP/s`
- `linear x[2048,512] @ W[2048,512]^T`: `21.786 ms`, `197 GFLOP/s`

Criterion baseline via RCH `vmi1227854`, commit `c0b6f3da`:

- `linear_forward/hidden/1024`: `[847.91 us 920.70 us 1.0068 ms]`
- `linear_forward/hidden/2048`: `[1.5089 ms 1.6456 ms 1.8207 ms]`

## Lever Tried

Pack each f64 `dgemm_bt` output-column panel into contiguous logical
`B^T[k,bj]` storage once per `j` tile, then reuse that panel across row tiles.
A guard kept the original strided path when only one row tile exists.

Note: the stale scratch tracker temporarily assigned this local attempt the
`frankentorch-kgs4.53` id, but remote `main` already used that id for the
parallel var/std dim-reduction lane. The pushed closeout uses slugged tracker
IDs to avoid colliding with remote bead state.

K remained unsplit in every candidate, so floating-point accumulation order
inside each matrixmultiply call matched the materialized-transpose reference.
Tile writes remained disjoint. No RNG, ordering, or tie-breaking behavior is
introduced by this path.

## Behavior Proof

Final guarded candidate passed:

- `rch exec -- cargo check -j 1 -p ft-kernel-cpu --lib --examples --benches`
  on `vmi1227854`; pre-existing `gemm_golden` example warnings only.
- `rch exec -- cargo test -j 1 -p ft-kernel-cpu dgemm_bt_matches_materialized_transpose_bit_exact -- --nocapture`
  on `vmi1227854`: `1 passed; 0 failed`.

The exactness test covers wide-column, row-split, and serial `dgemm_bt` gates
against a materialized-transpose `dgemm` reference using bit equality.

## Benchmark Outcome

Same-worker paired `gemm_bt_ab` on RCH `vmi1149989`:

| Shape | Baseline | Final Candidate | Speedup |
| --- | ---: | ---: | ---: |
| `[512,1024] x [1024,1024]^T` | `6.162 ms` | `6.752 ms` | `0.91x` |
| `[1024,1024] x [1024,1024]^T` | `14.270 ms` | `14.665 ms` | `0.97x` |
| `[2048,512] x [2048,512]^T` | `29.455 ms` | `33.362 ms` | `0.88x` |

Same-worker paired `gemm_bt_ab` on RCH `vmi1264463`:

| Shape | Baseline | Final Candidate | Speedup |
| --- | ---: | ---: | ---: |
| `[512,1024] x [1024,1024]^T` | `33.164 ms` | `27.782 ms` | `1.19x` |
| `[1024,1024] x [1024,1024]^T` | `48.772 ms` | `75.386 ms` | `0.65x` |
| `[2048,512] x [2048,512]^T` | `117.651 ms` | `81.574 ms` | `1.44x` |

The only clean final-source paired results are either all-regression
(`vmi1149989`) or mixed with a severe middle-shape regression (`vmi1264463`).
Score is below the keep threshold:

- Impact: not consistently positive; geometric mean is below `1.0x` on
  `vmi1149989` and only about `1.04x` on noisy `vmi1264463`.
- Confidence: low, because the final-source candidate did not reproduce the
  earlier unguarded cross-run win under paired evidence.
- Effort/risk: moderate, because the lever adds packing allocation/copy work
  to a hot kernel path.

`Impact x Confidence / Effort < 2.0`; source change was reverted manually.

## Next Primitive

Do not continue this panel-copy micro-family. The next `dgemm_bt` attack should
replace the per-call packing idea with an algorithmically deeper primitive:

- transpose-aware microkernel traversal that reads B rows with cache-friendly
  K-major blocking without allocating panels, or
- persistent packed-weight cache at Linear/module construction boundaries with
  explicit invalidation/versioning, so packing cost is amortized across many
  forwards/backwards.
