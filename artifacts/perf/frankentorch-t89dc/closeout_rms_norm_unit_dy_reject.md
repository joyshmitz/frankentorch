# frankentorch-t89dc closeout: RMSNorm sum-gradient stat staging rejected

Target:
- `rms_norm/grad_2048x1024`
- Profile source: `artifacts/perf/frankentorch-next-reprofile-20260617c/current_top_train_reprofile_after_16m8a.log`
- Reprofile timing: `[146.98 ms 150.93 ms 154.95 ms]`

Lever tried:
- Add an exact f64 RMSNorm all-ones upstream-gradient path that stages per-row `rstd` once inside backward and reuses it for `dx` plus serial `dweight`.
- The candidate source was removed because the benchmark did not provide a significant win.

Behavior proof during trial:
- Kernel bit proof passed against the old recomputing formula for `dx` and `dweight`.
- RMSNorm API finite-difference and golden-bit gradient tests passed.
- Candidate golden digest: `0x32b168a7f76dce91`.
- Ordering/RNG/isomorphism notes: no RNG or tie behavior; `dweight` retained old serial row-major accumulation order; non-one upstream gradients fell through unchanged.

Benchmark result:
- Baseline: `[130.96 ms 145.46 ms 159.22 ms]`
- Candidate: `[131.89 ms 140.44 ms 149.79 ms]`
- Criterion change: `[-14.089% -3.4562% +8.7184%]`, `p = 0.58`
- Decision: reject. Median moved in the right direction but confidence was too low and intervals overlapped.
- Score: `0.00`

Next route:
- Do not repeat norm stat-staging micro-levers.
- Move to a different structural primitive, such as pooling backward scatter/elision or SDPA backward memory traffic.
