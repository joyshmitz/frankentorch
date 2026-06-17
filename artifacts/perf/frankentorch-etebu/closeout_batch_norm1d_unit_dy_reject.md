# frankentorch-etebu closeout: BatchNorm1d all-ones-dy route rejected

Target:
- `batch_norm/grad_1d_8192x1024`
- Profile source: `artifacts/perf/frankentorch-next-reprofile-20260617b/current_top_train_reprofile_after_6olvt.log`
- Reprofile timing after `frankentorch-6olvt`: `[673.99 ms 685.79 ms 699.70 ms]`

Lever tried:
- Extend the existing f64 BatchNorm all-ones upstream-gradient fast path to `spatial == 1`.
- Fallback behavior for non-one `dy` stayed on the original branch.
- The candidate source change was removed because the benchmark did not clear the keep threshold.

Behavior proof during trial:
- Kernel bit proof against old spatial1 serial reference passed:
  `cargo test -j 1 -p ft-kernel-cpu 'batch_norm_f64_spatial' -- --nocapture`
- API gradient checks passed:
  `cargo test -j 1 -p ft-api 'functional_batch_norm1d_grad' -- --nocapture`
- Spatial1 candidate golden digest observed during the rejected trial:
  `0x59cbd999765bf72a`
- Ordering/tie/RNG/isomorphism notes:
  no RNG or tie-breaking is introduced; the guard was exact `dy.to_bits() == 1.0f64.to_bits()`; non-one upstream gradients fall back to the original path; candidate reference comparisons used the old channel-major dweight/dbias and row-major dx formulas.

Benchmark result:
- Baseline: `[701.01 ms 713.15 ms 727.28 ms]`
- Candidate: `[690.70 ms 705.03 ms 719.06 ms]`
- Criterion change: `[-3.9400% -1.1400% +1.5032%]`, `p = 0.44`
- Decision: reject. Median speedup was only `1.0115x` and not statistically significant.
- Score: `0.25 = 1.0115 impact * 0.50 confidence / 2.0 effort`

Next route:
- Do not repeat spatial1 BatchNorm all-ones micro-specialization.
- Move to a structurally different primitive from the current profile, such as fusing GroupNorm/LayerNorm backward statistics or replacing serial affine-gradient recomputation with an order-preserving staged algorithm.
