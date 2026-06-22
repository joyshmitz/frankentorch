# frankentorch-kgs4.161 — batched linalg.inv (extends kgs4.160 batched-solve kernel)

Date: 2026-06-21
Agent: cc — frankentorch-qe48n follow-up

## Lever

`tensor_linalg_inv` was 2-D only (3-D error). A^{-1} = solve(A, I): added a batched no-grad f64
fast path that builds a per-matrix identity RHS and calls the kgs4.160
`lu_solve_batched_contiguous_f64` kernel. No new kernel — reuses the chunked-parallel batched
LU-solve. Matches torch's batched inv (was a parity gap).

## Correctness: bit-exact-to-tolerance vs PyTorch (probe rel `1.7e-12`–`4.6e-12`); ft-api `--lib linalg_inv` 3 passed; ft-conformance green.

## Measurement (same-host, no-grad batched inv, 32 threads, f64)

| Shape | FT | PyTorch | verdict |
| --- | ---: | ---: | --- |
| `[100000,4,4]` | `8.2 ms` | `10.0 ms` | **FT 1.22x faster** |
| `[20000,16,16]` | `77.7 ms` | `47.6 ms` | FT 1.63x slower |
| `[5000,32,32]` | `25.7 ms` | `99.7 ms` | **FT 3.88x faster** |

## Verdict: KEEP — parity feature + perf win at most sizes (2W / 1L)

Same profile as batched solve (kgs4.160): adds batched inv (was erroring), wins small/large k,
loses at PyTorch's well-tuned k=16 gesv/getri sweet spot. No regression. The k=16 gap is FT's
scalar small-LU vs LAPACK (a SIMD/blocked-small-LU lever). Remaining qe48n: det/cholesky (PyTorch
batches those efficiently — 0.96-17.9ms — so lower-EV).

## Gates
- `cargo test -p ft-api --release --lib linalg_inv`: 3 passed.
- `cargo test -p ft-conformance --release`: green.
