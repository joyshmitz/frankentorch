# frankentorch-g047y closeout: AvgPool2d 2x2s2 backward rejected

Target:
- `avg_pool2d/grad`
- Profile source: `artifacts/perf/frankentorch-next-reprofile-20260617c/current_top_train_reprofile_after_16m8a.log`
- Reprofile timing: `[104.25 ms 106.38 ms 108.83 ms]`

Lever tried:
- Add a f64 `avg_pool2d_backward_f64` branch for 2x2 kernel, stride 2, no padding, `count_include_pad=true`.
- The branch preserved exact generic semantics by using `+= dout / 4.0` for each non-overlapping input cell.
- The candidate source was removed because the benchmark did not improve.

Behavior proof during trial:
- Kernel bit proof passed against the generic backward formula, including `-0.0` upstream gradient coverage.
- API finite-difference AvgPool2d gradient test passed.
- Candidate golden digest: `0xa99a68127e9e89a5`.
- Ordering/RNG/isomorphism notes: no RNG or tie behavior; output-window scan order and per-cell `+=` semantics matched the generic path; only the exact 2x2 stride2 no-padding geometry was guarded.

Benchmark result:
- Baseline: `[65.864 ms 66.725 ms 67.963 ms]`
- Candidate: `[65.705 ms 67.269 ms 68.601 ms]`
- Criterion change: `[-4.6956% -0.1177% +5.3235%]`, `p = 0.96`
- Decision: reject. No significant improvement.
- Score: `0.00`

Next route:
- Move to a larger structural bottleneck rather than 2x2 pooling backward simplification; SDPA backward memory traffic or a deeper BatchNorm algorithmic rewrite remain candidates.
