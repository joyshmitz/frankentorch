# Perf Profiling Pass — Scenario & Success Metrics

- **Run ID:** 20260601T2325Z-rustickite
- **Owner:** RusticKite (claude-code / opus-4.8)
- **Date:** 2026-06-01
- **Mode:** MEASUREMENT ONLY. No optimization in this pass. Top hotspots handed off to optimizer agents via `perf`-tagged beads.

## What "fast" means here

FrankenTorch is a CPU-first, single-threaded-by-default tensor library aiming for PyTorch parity. The realistic workload is **per-op and per-layer latency on the hot training/inference path** at training-typical tensor sizes. We measure the operators that dominate a transformer/MLP step:

| Scenario | Op | Sizes | Why representative |
|----------|----|-------|--------------------|
| S1 matmul | `tensor_matmul` (GEMM) | square 64/128/256/512/1024 | core of every linear/attention |
| S2 linear fwd | `tensor_linear` (transpose+addmm+bias) | batch=32, in=512, hidden 256/512/1024/2048 | MLP/projection layer; worst prior ratio (10–12× torch) |
| S3 conv2d | `tensor_conv2d` | 1×3×{64,128}² img, 16×3×3×3 weight | vision hot path; prior 4–6× torch |
| S4 elementwise | `relu`, `exp`, `add` | 1M elements | activation/residual; relu+add were 8–30× torch pre-SIMD |
| S5 reductions | `sum`, `softmax` | 1M / vocab 1k–50k | loss + attention/softmax |
| S6 backward | `backward_matmul` | 64/128/256 | autograd tape replay cost |

## Success metric

- **Primary:** p50/p95 wall-clock latency per op (criterion, ≥20 samples — criterion default 100).
- **Secondary:** throughput (elements/sec) and peak RSS (`/usr/bin/time -v`) for the elementwise 1M case.
- **Comparison anchor:** `benchmarks/pytorch_baseline.json` (PyTorch 2.12.0+cpu, single-threaded) from prior pass — used only to rank gaps, not as the pass/fail gate.

## Budget / target (for the optimizer agents who pick up the beads)

A FrankenTorch op is "acceptable" at ≤1.5× the single-threaded PyTorch p50, "needs work" at 1.5–3×, "perf bug" at >3×. The beads filed from this pass target the >3× rows first.

## Build & execution discipline

- Build profile: `bench` (inherits `release`) + `RUSTFLAGS=-C force-frame-pointers=yes` for profiler attribution (frame pointers on, `strip=false`). Not the size-optimized default release.
- **Compilation** offloaded via `rch` (shared build farm); **benchmarks executed locally** on the fingerprinted host so timings are host-consistent.
- One lever per run; ≥20 samples; p50/p95 reported.

## Known caveats (recorded, not corrected)

1. **CPU governor = `powersave`** (not `performance`) and `no_turbo` n/a — absolute numbers carry P-state jitter; treat ratios/attribution as primary, absolute µs as indicative. (Skill rule: ASK before tuning kernel/governor — not tuned here.)
2. **Shared host with peer agents** — not isolated; expect wider variance than the ≤10% envelope. Flag any op with >20% p95 drift across runs.
3. **`bench_linear_forward` reuses one session across `b.iter()`** — each `tensor_linear` call appends nodes to the autograd tape, so tape size grows during the measurement. This is itself a hypothesis (tape-growth overhead) recorded in the ledger, not silently corrected.
