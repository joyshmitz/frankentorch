# FrankenTorch Performance SLO Baseline V1

This is the committed baseline artifact for the perf-SLO runner used by the future G5 wiring.

- source: direct worker-captured full-profile bundle assembled from four 25-iteration remote shards on `2026-04-09` UTC
- profile: `full`
- iterations: `100`
- warmup_iterations: `1`
- versioned report json: `artifacts/phase2c/performance/perf_slo_report_full_100x_v1.json`
- versioned summary md: `artifacts/phase2c/performance/PERF_SLO_SUMMARY_FULL_100X_V1.md`
- canonical baseline json: `artifacts/phase2c/performance/perf_slo_baseline_v1.json`
- canonical baseline summary: `artifacts/phase2c/performance/PERF_SLO_BASELINE_V1.md`
- shard inputs: `shard01` from `vmi1153651`, `shard02` from `vmi1152480`, `shard03` from `vmi1293453`, `shard04` from `vmi1264463`

This replaces the prior single-sample placeholder anchor. The bundle preserves the generated shard artifacts directly and widens the statistical envelope to the intended 100 measured iterations plus one warmup.

Observed direct spec-budget status from this baseline:

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
