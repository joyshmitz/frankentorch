# REJECTED: parallelizing the eigh tred2 reduction + back-transform (2026-06-14, BlackThrush)

## Context
After shipping the eigh **tql2** eigenvector sweep via deferred whole-stream replay
(kgs4.73 f64 3.1-4.6x, kgs4.74 f32 3.9-4.3x), the remaining serial phases of
`eigh_contiguous_f64` are the Householder **tridiagonalization** (`eigh_tred2_reduce_packed_full`)
and the eigenvector **back-transform** (`eigh_tred2_backtransform`).

Profiling (n=768): reduce ~374ms, back-transform ~124ms, tql2-region (now parallel)
~555ms of ~1053ms total. So reduce+back-transform ≈ 47% of current eigh.

## Lever attempted
Both phases' inner trailing updates ARE independent per output column/row and
parallelize **bit-exactly** (verified max|ΔV|=max|Δλ|=0.0):
- reduce matvec `e=(A·row_i)/h`: `par_iter_mut` over `e` (safe).
- reduce symmetric rank-2 update of the packed lower triangle: a new gemm raw-ptr
  helper `eigh_tred2_rank2_update_f64` (disjoint packed rows), after pre-finalizing `e`.
- back-transform projection (GEMV) + apply (rank-1): `par_iter_mut` / `par_chunks_mut`.

## Result: REJECTED (regression)
Same-process A/B (10 threads, f64 eigh with vectors; serial-reduce vs all-parallel):
  n=512  205.3 -> 292.8 ms  **0.70x (REGRESSION)**
  n=768  829.1 -> 766.2 ms  1.08x
bit-exact (max|ΔV|=max|Δλ|=0.0) but SLOWER.

## Why
The tridiagonalization/back-transform outer reflector loop is **inherently
sequential** — reflector `i` modifies the matrix that reflector `i-1` reads, so it
CANNOT be deferred (unlike tql2, whose d/e recurrence never touches the vectors).
Parallelizing only WITHIN each reflector means ~n separate rayon fork/joins, each
with O(i) work; the join-barrier + thread-wakeup overhead swamps the gain at the
benched sizes. This is the same "O(n) fork/joins" anti-pattern that regressed the
per-rotation SVD/tql2 attempts.

Even if it had NOT regressed: tql2 (already parallel) is ~50% of current eigh, so
fully parallelizing the other ~50% caps total eigh at ~1.7-1.8x by Amdahl — below
the Score>=2.0 bar anyway.

## NEXT TARGET (named per the no-ceiling reporting rule)
**Non-symmetric eig eigenvector back-substitution** (`eig_backsub_eigenvectors`,
EISPACK hqr2): each eigenvector (column `en` of the Schur form) is an independent
back-solve of (T-λI)x=0 → embarrassingly parallel OVER eigenvalues (ONE fork/join,
not per-reflector). Currently computed IN PLACE over H (column en reads Schur
columns < en that later iterations overwrite → false serial dependency); the fix is
to write eigenvectors into a SEPARATE buffer reading the intact Schur H, then the
existing q_acc GEMM transform. Non-sym eig is the biggest remaining vs-LAPACK gap
(12-40x). Target: parallelize-over-eigenvector, ~3-8x on the back-sub phase.
