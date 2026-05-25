# FrankenTorch Profiling Results

Rigorous performance comparison against PyTorch 2.12.0+cpu (single-threaded).

## Environment

| Component | Version |
|-----------|---------|
| PyTorch | 2.12.0+cpu |
| FrankenTorch | 0.1.0 |
| Platform | Linux 6.17.0 x86_64 |
| CPU | (TBD after benchmark) |
| Threads | 1 (single-threaded) |

## Benchmark Methodology

- **Warmup**: 10 iterations discarded
- **Runs**: 30 timed iterations per benchmark
- **Metrics**: p50, p95, p99 latency in microseconds
- **Apples-to-apples**: Both libraries run single-threaded on CPU

## Results Summary

| Operation | Size | PyTorch p50 (μs) | FrankenTorch p50 (μs) | Ratio | Status |
|-----------|------|------------------|----------------------|-------|--------|
| matmul | 64x64 | 9.02 | TBD | TBD | ⏳ |
| matmul | 128x128 | 49.05 | TBD | TBD | ⏳ |
| matmul | 256x256 | 365.15 | TBD | TBD | ⏳ |
| matmul | 512x512 | 2908 | TBD | TBD | ⏳ |
| matmul | 1024x1024 | 24022 | TBD | TBD | ⏳ |
| bmm | b8 128x128 | 390 | TBD | TBD | ⏳ |
| bmm | b16 128x128 | 845 | TBD | TBD | ⏳ |
| bmm | b32 128x128 | 23124 | TBD | TBD | ⏳ |
| conv2d | 32x32 | 3582 | TBD | TBD | ⏳ |
| conv2d | 64x64 | 28158 | TBD | TBD | ⏳ |
| conv2d | 128x128 | 73183 | TBD | TBD | ⏳ |
| sum | 1M | 48.79 | TBD | TBD | ⏳ |
| softmax | 32x8192 | 292.73 | TBD | TBD | ⏳ |
| relu | 1M | 91.19 | TBD | TBD | ⏳ |
| exp | 1M | 1182 | TBD | TBD | ⏳ |
| add | 1M | 110.34 | TBD | TBD | ⏳ |
| backward_matmul | 64x64 | 140.21 | TBD | TBD | ⏳ |
| backward_matmul | 128x128 | 387.75 | TBD | TBD | ⏳ |
| backward_matmul | 256x256 | 1819 | TBD | TBD | ⏳ |
| linear | 32x512→256 | 113.93 | TBD | TBD | ⏳ |
| linear | 32x512→1024 | 440.56 | TBD | TBD | ⏳ |
| linear | 32x512→2048 | 885.89 | TBD | TBD | ⏳ |

## Status Legend

- ✅ FrankenTorch ≤ 1.5x PyTorch (acceptable)
- ⚠️ FrankenTorch 1.5x-3x PyTorch (needs optimization)
- ❌ FrankenTorch > 3x PyTorch (performance bug filed)
- ⏳ Benchmark in progress

## Performance Gaps (beads filed)

TBD after benchmarks complete.

## Optimization Opportunities

TBD after profiling with flamegraphs.

---
Generated: 2026-05-25
Benchmark in progress...
