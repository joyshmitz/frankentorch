# frankentorch-g1upz.1 rejection report

## Target

- Bead: `frankentorch-g1upz.1`
- Primitive: packed B 4-column panels plus a 4-row register tile using safe `std::simd` fused `mul_add`.
- Scope: f64 `512x512x512` GEMM route in `crates/ft-kernel-cpu/src/lib.rs`.

## Baseline

- Valid same-worker baseline for final comparison:
  - Worker: `vmi1156319`
  - Log: `artifacts/perf/frankentorch-g1upz/baseline_gemm_512_vmi1156319.log`
  - `matmul_f64_512x512x512`: `[11.015 ms 13.464 ms 17.091 ms]`
- Additional pre-change baseline captured on `vmi1153651`:
  - Log: `artifacts/perf/frankentorch-g1upz-1/baseline_dgemm512.log`
  - `matmul_f64_512x512x512`: `[18.759 ms 23.369 ms 31.277 ms]`
- Failed fresh-baseline attempt:
  - Log: `artifacts/perf/frankentorch-g1upz/baseline_child_packed_register_512.log`
  - RCH refused local fallback before assigning a worker.

## Lever

- One lever implemented and tested, then removed after the score gate failed.
- Shape gate: f64 `512x512x512`.
- B layout: packed 4-column panels.
- Register tile: four output rows by four columns.
- FP contract: lane-wise fused `mul_add` via safe `std::simd::StdFloat`, whose implementation calls non-relaxed `simd_fma`.
- Ordering ledger: output rows/columns independent; inner accumulation order remained ascending `p = 0..k`; no tie-breaking or RNG involved.

## Proof

- Focused bit-order proof passed remotely:
  - Worker: `vmi1153651`
  - Log: `artifacts/perf/frankentorch-g1upz/test_packed_register_simd_fma_retry2.log`
  - Result: `1 passed; 0 failed`
- Golden SHA after source removal:
  - Log: `artifacts/perf/frankentorch-g1upz-1/golden_sha256_after_rejection.log`
  - Result: all listed golden outputs `OK`.

## After Benchmark

- Worker: `vmi1156319`
- Log: `artifacts/perf/frankentorch-g1upz-1/after_packed_register_simd_fma.log`
- `matmul_f64_512x512x512`: `[215.87 ms 232.03 ms 250.63 ms]`

## Score

- Same-worker median comparison on `vmi1156319`: `13.464 ms -> 232.03 ms`
- Ratio: `0.0580x` (`17.23x` slower)
- Score: `0.0`
- Verdict: REJECTED; source hunk removed.

## Next Primitive

Do not repeat scalar row-major replay, packed 4-column scalar tiles, or safe `std::simd` FMA tiles for this target. The next child should attack a different primitive: a safe-Rust software-FMA microkernel/artifact that avoids the slow `simd_fma` fallback path, then re-evaluate packed macro-panels only after that primitive proves faster than the rejected `232 ms` class.
