# ft-kernel-cpu GEMM row-block parallelism threshold

- Target: `gemm::dgemm` / `gemm::sgemm` row-block parallel gate (`PAR_MIN_FLOPS`)
- Lever: lower `PAR_MIN_FLOPS` from `1<<29` (~537M FMA) to `1<<27` (~134M FMA)
- Files: `crates/ft-kernel-cpu/src/lib.rs`

## Profile Target

`should_parallelize` only split a GEMM across the rayon pool above `1<<29`
(~537M) fused multiply-adds. On a 64-core worker that left two important
medium GEMMs running single-threaded while 63 cores sat idle:

- the conv2d im2col matmul `[M=4096, K=576] x [K=576, N=64]` ≈ 151M FMA
  (`ops_bench conv2d/hw/32`), and
- a `512x512` square matmul ≈ 134M FMA (`ops_bench matmul/square/512`).

Both fall in the band `[134M, 537M)` that was gated to serial. The
`matmul/square/256` case (16.7M) and everything smaller stays serial.

## Lever

One constant: `PAR_MIN_FLOPS: 1<<29 -> 1<<27`. Nothing else changes. Sizes in
`[134M, 537M)` now take the existing `c.par_chunks_mut(...).zip(a.par_chunks(...))`
row-block path; smaller GEMMs are untouched.

## Isomorphism Proof

- The row-block split feeds each block to the same `matrixmultiply` micro-kernel
  over a contiguous row range of A and C. For a given output element the
  k-accumulation order is fixed by the micro-kernel and does NOT depend on the
  block/row count, so the parallel result is bit-for-bit identical to the single
  call. Proved by the existing `gemm_row_split_matches_single_bit_exact` test
  (re-run green under the lowered threshold).
- Ordering / tie-breaking / dtype / RNG: N/A — pure dense matmul, no reordering.
- conv2d / matmul focused tests pass under the new threshold.

## Result (same 64-core worker, RAYON_NUM_THREADS=1 vs default)

Measured back-to-back on ONE worker to remove cross-worker variance (identical
HEAD code was seen to swing 32.7ms<->47.3ms across workers, so cross-worker A/B
is unreliable). `RAYON_NUM_THREADS=1` reproduces the old serial behaviour since
`should_parallelize` requires `current_num_threads() > 1`.

```text
                       serial (=old)      parallel (=new)     speedup
matmul/square/512      6.8693 ms          2.7381 ms           2.51x
conv2d/hw/32           30.629 ms          21.817 ms           1.40x
```

`matmul/square/512` parallel (2.738 ms) now matches/beats the PyTorch 2.12 CPU
anchor `matmul_512x512` p50 2.908 ms — the gap is closed. conv2d/hw/32 gains
1.40x; its remaining cost is the serial im2col gather, not the matmul.

## Opportunity Score

Impact 4 x Confidence 5 / Effort 1 = 20.0. Bit-exact, one constant, large
same-worker win on the dominant medium-GEMM band.

## Next Primitive

conv2d's residual gap is the serial im2col panel materialization (the hw/128
panel is ~302 MB of pure gather traffic). The next deeper swing is a blocked /
implicit im2col-GEMM that tiles the output so each tile's panel stays in L2,
fusing the gather with the matmul and avoiding the full-panel allocation.
