# FrankenTorch Performance SLO Summary

- schema_version: `ft-perf-slo-report-v1`
- generated_unix_ms: `1775699967127`
- profile: `full`
- iterations: `100`
- warmup_iterations: `1`
- overall_status: `fail`

| Budget | Status | Observed | Limit | Detail |
|---|---|---|---|---|
| Elementwise forward | fail | 3873.502 p95_ms | 200.000 p95_ms | observed metric exceeds the spec budget |
| Backward pass | fail | 3.072 ratio | 1.350 ratio | observed metric exceeds the spec budget |
| Matmul core | fail | 27918.526 p95_ms | 240.000 p95_ms | observed metric exceeds the spec budget |
| Optimizer step | fail | 1812.267 p95_ms | 130.000 p95_ms | observed metric exceeds the spec budget |
| Checkpoint save/load | fail | 76235.622 p95_ms | 6135.668 p95_ms | compares the slower of save/load p95 against the latency implied by 350 MB/s |
| Memory footprint | baseline_required | 38500.000 pct | 8.000 pct | peak RSS regression versus baseline training-step profile; provide --baseline to evaluate regression |
| Allocation churn | baseline_required | 25400.000 pct | 10.000 pct | allocation-op regression versus baseline backward-heavy trace; provide --baseline to evaluate regression |
| Tail stability | baseline_required | 29277310050.000 pct | 7.000 pct | family p99 regression versus baseline benchmark family; provide --baseline to evaluate regression |
