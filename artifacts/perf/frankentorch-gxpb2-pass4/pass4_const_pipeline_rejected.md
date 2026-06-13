# frankentorch-gxpb2 pass 4 const pipeline rejection

Bead: `frankentorch-gxpb2`
Lever tried: const-specialize `eig_impl` and `hessenberg_reduce_blocked` over the
values-only vs vector-producing path, routing public `eigvals` through the
`false` instantiation and public `eig` through the `true` instantiation.

## Baseline

Command:

`RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench eigvals_f64_256x256 -- --warm-up-time 1 --measurement-time 3 --sample-size 10`

Actual worker selected: `vmi1152480`.

`eigvals_f64_256x256` baseline:

`[28.116 ms 28.944 ms 29.404 ms]`

## Candidate

Command:

`RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1152480 rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench eigvals_f64_256x256 -- --warm-up-time 1 --measurement-time 3 --sample-size 10`

Actual worker selected: `vmi1152480`.

`eigvals_f64_256x256` candidate:

`[27.649 ms 29.116 ms 30.479 ms]`

Median result: `29.116 / 28.944 = 1.006x` slower. This fails the Score gate.

## Behavior proof

- Ordering and complex-pair slots: the candidate did not change shift selection,
  selected-`m`, active-window search, deflation order, or eigenvalue slot layout.
- Floating point: the candidate only hoisted `want_vectors` to const-generic
  dispatch; arithmetic expressions stayed in the same source order.
- RNG: none.
- Strict golden while the candidate was present still produced extracted stdout
  SHA-256 `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.
- The source hunk was removed after the failed benchmark; `git diff --
  crates/ft-kernel-cpu/src/lib.rs` is empty for this closeout.

## Verdict

Rejected. Source unchanged.

Score: `0.0` because the same-worker median regressed.

Next route: do not keep widening const-specialization micro-levers on this
family. Continue `gxpb2` through a deeper primitive: grouped/far-update operation
tape for BLAS-3-style Francis updates, or a strict-fallback Schur-window/AED
kernel that can prove shift stream, selected-`m`, deflation counters, Schur
bits, eigenvalue slots, and strict golden SHA before public dispatch.
