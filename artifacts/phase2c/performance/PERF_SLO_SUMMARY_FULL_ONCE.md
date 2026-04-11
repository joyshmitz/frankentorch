# FrankenTorch Performance SLO Summary

- schema_version: `ft-perf-slo-report-v1`
- generated_unix_ms: `1775595086784`
- profile: `full`
- iterations: `1`
- warmup_iterations: `0`
- overall_status: `fail`

| Budget | Status | Observed | Limit | Detail |
|---|---|---|---|---|
| Elementwise forward | fail | 1412.540 p95_ms | 200.000 p95_ms | observed metric exceeds the spec budget |
| Backward pass | fail | 2.741 ratio | 1.350 ratio | observed metric exceeds the spec budget |
| Matmul core | fail | 10074.619 p95_ms | 240.000 p95_ms | observed metric exceeds the spec budget |
| Optimizer step | fail | 635.020 p95_ms | 130.000 p95_ms | observed metric exceeds the spec budget |
| Checkpoint save/load | fail | 27852.424 p95_ms | 6135.668 p95_ms | compares the slower of save/load p95 against the latency implied by 350 MB/s |
| Memory footprint | baseline_required | 14732.000 pct | 8.000 pct | peak RSS regression versus baseline training-step profile; provide --baseline to evaluate regression |
| Allocation churn | baseline_required | 254.000 pct | 10.000 pct | allocation-op regression versus baseline backward-heavy trace; provide --baseline to evaluate regression |
| Tail stability | baseline_required | 10074618909.000 pct | 7.000 pct | family p99 regression versus baseline benchmark family; provide --baseline to evaluate regression |
