# FrankenTorch Performance SLO Summary

- schema_version: `ft-perf-slo-report-v1`
- generated_unix_ms: `1775795479414`
- profile: `full`
- iterations: `100`
- warmup_iterations: `1`
- overall_status: `fail`
- baseline_report: `artifacts/phase2c/performance/perf_slo_baseline_v1.json`

| Budget | Status | Observed | Limit | Detail |
|---|---|---|---|---|
| Elementwise forward | fail | 1537.970 p95_ms | 200.000 p95_ms | observed metric exceeds the spec budget |
| Backward pass | fail | 2.686 ratio | 1.350 ratio | observed metric exceeds the spec budget |
| Matmul core | fail | 10723.378 p95_ms | 240.000 p95_ms | observed metric exceeds the spec budget |
| Optimizer step | fail | 643.857 p95_ms | 130.000 p95_ms | observed metric exceeds the spec budget |
| Checkpoint save/load | fail | 26425.495 p95_ms | 6135.668 p95_ms | compares the slower of save/load p95 against the latency implied by 350 MB/s |
| Memory footprint | pass | 15036.000 pct | 8.000 pct | peak RSS regression versus baseline training-step profile |
| Allocation churn | pass | 25400.000 pct | 10.000 pct | allocation-op regression versus baseline backward-heavy trace |
| Tail stability | pass | 10921606148.000 pct | 7.000 pct | family p99 regression versus baseline benchmark family |
