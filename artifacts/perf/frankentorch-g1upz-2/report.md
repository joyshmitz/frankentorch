# frankentorch-g1upz.2 — row-parallel software-FMA GEMM rejection

## Verdict

Rejected. The row-parallel safe-Rust software-FMA microkernel preserved the deterministic ascending-`p` `f64::mul_add` contract, but it did not clear the performance gate.

## Baseline

- Primary child baseline: `artifacts/perf/frankentorch-g1upz-2/baseline_software_fma_dgemm512.log`
- Worker: `vmi1167313`
- Target: `cargo bench -p ft-kernel-cpu --bench gemm_bench -- matmul_f64_512x512x512`
- Time: `[6.8222 ms 7.3593 ms 7.6842 ms]`

The after-benchmark was assigned to `vmi1156319`; the comparable pre-edit same-worker baseline is the existing no-code-diff GEMM baseline from the previous pass:

- Same-worker baseline: `artifacts/perf/frankentorch-g1upz/baseline_gemm_512_vmi1156319.log`
- Worker: `vmi1156319`
- Time: `[11.015 ms 13.464 ms 17.091 ms]`

## Lever

One temporary source lever was tested and then removed:

- Route only f64 `512x512x512` GEMM through a safe row-parallel software-FMA helper.
- Each output row was exclusively owned by one Rayon task.
- Each output element accumulated `p = 0..k` in ascending order with `f64::mul_add`.
- Columns were processed eight at a time to reuse the loaded `A[i,p]` value and avoid `std::simd::simd_fma`.

## Proof

- Compile check: `artifacts/perf/frankentorch-g1upz-2/check_row_parallel_software_fma.log`
- Focused proof: `artifacts/perf/frankentorch-g1upz-2/test_row_parallel_software_fma_order_retry1.log`
- Golden SHA after source removal: `artifacts/perf/frankentorch-g1upz-2/golden_sha256_after_reject.log`

Isomorphism ledger:

- Ordering: row ownership independent; for every output element, `p` accumulation order is exactly ascending.
- Tie-breaking: not applicable.
- Floating point: temporary helper used explicit `f64::mul_add`; no reassociation within an output element.
- RNG: not applicable.
- Final tree state: source hunk removed; `crates/ft-kernel-cpu/src/lib.rs` has no diff.

## Re-benchmark

- After log: `artifacts/perf/frankentorch-g1upz-2/after_row_parallel_software_fma_vmi1167313.log`
- Actual after worker: `vmi1156319`
- Time: `[226.63 ms 251.31 ms 274.17 ms]`

Comparable same-worker ratio:

- `13.464 ms -> 251.31 ms`
- Speed ratio: `0.0536x`
- Regression: `18.66x` slower
- Score: `0.0`

Cross-worker child-baseline ratio:

- `7.3593 ms -> 251.31 ms`
- Speed ratio: `0.0293x`
- Regression: `34.15x` slower

## Next Primitive

Do not repeat scalar row-major, row-parallel software-FMA, packed B panel, or `std::simd::simd_fma` fallback loops. The next child should attack a fundamentally different deterministic primitive: a Kulisch/ExBLAS-style binned accumulator tile with a proofed rounding contract and an explicit target of at least `10x` faster than the `251 ms` software-FMA class before comparing against matrixmultiply baselines.
