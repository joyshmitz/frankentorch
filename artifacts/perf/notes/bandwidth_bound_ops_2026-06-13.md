# Perf investigation: the parallelization frontier is now memory-bandwidth-bound

Date: 2026-06-13  Agent: BlackThrush  Worker: 8-10 thread rch nodes

## Context
The dim-reduction family (argmax/argmin/max/min/sum/mean/prod/var/std) is fully
parallelized (kgs4.50-53, 5-29x wins). gather is parallelized (kgs4.54, 2.5x).
This session probed the next serial ops. ALL turned out to be memory-bandwidth /
already-efficient-serial bound, so per-lane/per-block parallelization gives <2x.
Verified with RIGOROUS same-process A/B (1-thread vs N-thread pool calling the
SAME kernel — NOT a hand-written serial baseline, which inflates the ratio).

## Rejected this session (do NOT re-attempt with plain parallelization)
- argmax/max dim=0 CACHE-BLOCKED row-streaming: 0.86x (REGRESSION). The strided
  gather is NOT the bottleneck (it's compute/branch-bound ~8 GB/s); the
  per-element NaN-freeze check + coarser band parallelism cost more than saved.
- index_select dim=0: parallel 0.95x (regression); copy_from_slice serial only
  1.13x. Pure row-copy = DRAM-bandwidth bound; parallel adds contention.
- scatter_add dim=1: parallel 1.18x. Read input+index+src, RMW output = bandwidth
  + per-element normalize_strict_index_value validation bound.
- cumprod dim=1: rigorous same-kernel A/B = 1.36x. Memory-bound (read+write 32MB);
  the 1-thread kernel is already efficient. NOTE: a peer previously REJECTED
  cumprod parallel (bead 66pe, measured 1.83x REGRESSION same-worker). Do not ship.

## LESSON
A serial op only yields >=2x from parallelization when its serial baseline is
COMPUTE-heavy (transcendental per element) or CACHE-HOSTILE in a way that
distributing fixes (the strided-gather reductions). Ops that are already a tight
contiguous copy/scan/accumulate are DRAM-bandwidth-bound: parallelism <2x. Always
do the 1-thread-vs-N-thread SAME-KERNEL A/B (ThreadPoolBuilder), never serial-fn
vs parallel-kernel (different code paths inflate the ratio).

## NEXT DEEP TARGETS (different primitive, not more parallelization)
1. SIMD-vectorized ORDER-INDEPENDENT reductions (max/min VALUE, amax): the
   compute-bound last-dim reductions are ~3x off torch due to torch's SIMD vs ft's
   scalar pairwise. max/min are associative+commutative -> a portable_simd f64x4
   lane-max + horizontal max is BIT-EXACT. Target ~2-3x on last-dim max-value.
   (sum is NOT bit-exact under SIMD reassociation -> tolerance-only.)
2. Quickselect for median/kthvalue/quantile: ft-api composes these via a FULL
   tensor_sort O(n log n); torch uses introselect O(n). select_nth_unstable on the
   no-grad path = different complexity class, ~10x for large tensors. (Grad path
   keeps sort.) Needs an ft-api no-grad branch + care.
