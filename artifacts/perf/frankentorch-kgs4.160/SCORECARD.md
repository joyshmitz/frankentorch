# frankentorch-kgs4.160 — batched linalg.solve (parity feature + perf win at most sizes)

Date: 2026-06-21
Agent: cc — implements frankentorch-qe48n

## Lever

`tensor_linalg_solve` was 2-D ONLY (3-D → ShapeMismatch), while PyTorch batches solve. Added
`ft_kernel_cpu::lu_solve_batched_contiguous_f64` (A=[batch,n,n], B=[batch,n,m]): one parallel
kernel, batch chunked across the pool with per-worker reused LU/pivot scratch (no per-matrix
alloc/validation overhead — the earlier ft-api per-matrix loop lost 2.2x to exactly that).
Inline LU-with-partial-pivoting + forward/back substitution per matrix. Wired into the no-grad
3-D fast path of `tensor_linalg_solve`.

## Correctness (bit-exact-to-tolerance)

`lu_solve_batched_matches_per_matrix_2d` asserts rel<1e-9 vs the per-matrix 2-D
`lu_factor`+`lu_solve` reference (both LU + partial pivot). Probe rel vs PyTorch `1e-12`.
ft-api `--lib linalg_solve` 4 passed; ft-conformance green (199 lib + bins).

## Measurement (same-host, no-grad batched solve, 32 threads, f64, example batched_solve_probe)

| Shape | FT | PyTorch | verdict |
| --- | ---: | ---: | --- |
| `[100000,4,4]` | `7.7 ms` | `10.3 ms` | **FT 1.35x faster** |
| `[20000,16,16]` | `81.5 ms` | `49.5 ms` | FT 1.65x slower |
| `[5000,32,32]` | `31.7 ms` | `96.7 ms` | **FT 3.05x faster** |

## Verdict: KEEP — parity feature + perf win at most sizes (2W / 1L)

This ADDS batched solve (was a 3-D ShapeMismatch — a torch-parity gap), so it's a capability
win + no regression (FT went from error → working). Perf: FT wins at small k (1.35x) and large
k (3.05x, where PyTorch's batched gesv scales badly); PyTorch's well-tuned gesv wins at its k=16
sweet spot (1.65x). Honest mixed perf, net-positive. The k=16 gap is FT's scalar LU vs LAPACK's
tuned mid-size gesv (a SIMD/blocked-small-LU lever, not parallelism — chunking + de-branching
didn't move it). Follow-up: extend the same batched kernel to inv/cholesky/det (qe48n).

## Gates
- `cargo test -p ft-kernel-cpu --release --lib lu_solve_batched_matches_per_matrix_2d`: pass.
- `cargo test -p ft-api --release --lib linalg_solve`: 4 passed.
- `cargo test -p ft-conformance --release`: green.
