# FrankenTorch Performance SLO Measurement V2

Fresh measurement captured on 2026-04-10 against the 100x baseline.

- profile: `full`
- iterations: `100`
- warmup_iterations: `1`
- baseline: `artifacts/phase2c/performance/perf_slo_baseline_v1.json`
- measurement: `artifacts/phase2c/performance/perf_slo_measurement_v2.json`

## Budget Results

| Budget | Status | Observed | Limit | Detail |
|---|---|---|---|---|
| Elementwise forward | **pass** | 152.472 p95_ms | 200.000 p95_ms | within spec budget |
| Backward pass | fail | 2.138 ratio | 1.350 ratio | observed metric exceeds the spec budget |
| Matmul core | fail | 3623.377 p95_ms | 240.000 p95_ms | observed metric exceeds the spec budget |
| Optimizer step | fail | 161.595 p95_ms | 130.000 p95_ms | observed metric exceeds the spec budget |
| Checkpoint save/load | fail | 11405.502 p95_ms | 6135.668 p95_ms | compares the slower of save/load p95 against the latency implied by 350 MB/s |
| Memory footprint | **pass** | -65.7% regression | 8.0% limit | peak RSS improved vs baseline |
| Allocation churn | **pass** | 0.0% regression | 10.0% limit | allocation-op stable vs baseline |
| Tail stability | **pass** | -82.7% regression | 7.0% limit | p99 tails improved vs baseline |

## Summary

**Overall Status: partial (4/8 budgets pass)**

### Improvements since baseline:
- Elementwise forward now passes (152ms vs baseline's 3873ms - 25x improvement)
- All three regression budgets (memory, allocation, tail) pass with baseline comparison

### Remaining failures:
- Backward pass: 2.14x vs 1.35x target (59% over budget)
- Matmul core: 3623ms vs 240ms target (15x over budget) - expected for pure Rust, no BLAS
- Optimizer step: 162ms vs 130ms target (24% over budget)
- Checkpoint save/load: 11.4s vs 6.1s target (86% over budget)

### Notes

The matmul budget failure is expected for a CPU-only pure Rust implementation without BLAS/LAPACK acceleration. The spec budget of 240ms was set based on optimized BLAS performance.

The checkpoint throughput budget assumes 350 MB/s which may require optimized serialization or parallel I/O.
