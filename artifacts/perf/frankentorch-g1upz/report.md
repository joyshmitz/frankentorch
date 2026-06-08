# frankentorch-g1upz deterministic mul_add GEMM pilot

## Target

- Bead: `frankentorch-g1upz`
- Hotspot: `ft-kernel-cpu` f64 GEMM backing 512x512x512 matmul.
- Lever tried: route only `dgemm(512, 512, 512)` to a safe-Rust scalar `f64::mul_add` row-major microkernel with four output columns unrolled.
- Graveyard route: numeric kernels plus communication-avoiding linear algebra; this pilot tested the smallest single-rounding kernel before moving to packed/register-blocked panels.

## Baseline

- `vmi1156319`, current pre-change benchmark: `artifacts/perf/frankentorch-g1upz/baseline_gemm_512_vmi1156319.log`
  - f64 512: `[11.015 ms 13.464 ms 17.091 ms]`
  - f32 512: `[3.3980 ms 3.6912 ms 4.0212 ms]`
- Same-worker f64 comparison baseline on `vmi1167313`: `artifacts/perf/frankentorch-qb1g2/baseline_gemm_f64_512_retry2.log`
  - f64 512: `[7.6685 ms 8.3931 ms 9.3803 ms]`

## Proof

- Focused bit-order proof: `artifacts/perf/frankentorch-g1upz/test_mul_add_reference.log`
  - Worker: `vmi1167313`
  - Result: `1 passed; 0 failed`
- Ordering preserved in the pilot: rows ascending, columns ascending, inner `p` ascending.
- Tie-breaking: not applicable to GEMM.
- Floating-point: pilot used `f64::mul_add` for every product-plus-accumulate in the tested helper.
- RNG: not applicable.
- Golden SHA: not run for closeout because the lever was rejected and removed from the source tree.

## Rebench

- After benchmark: `artifacts/perf/frankentorch-g1upz/after_mul_add_512_vmi1156319.log`
  - Actual worker: `vmi1167313`
  - f64 512: `[204.07 ms 232.25 ms 262.64 ms]`
  - f32 512: `[3.0084 ms 3.2472 ms 3.5554 ms]`

## Verdict

- Rejected. Same-worker f64 median regressed from `8.3931 ms` to `232.25 ms` (`0.036x`, about `27.7x` slower).
- Score: `0.0`, below the `>= 2.0` keep gate.
- Source hunk removed after rejection; `git diff -- crates/ft-kernel-cpu/src/lib.rs` is empty.

## Next Primitive

Do not repeat scalar row-major `mul_add` replay. The next profile-backed swing should build a packed/register-blocked deterministic GEMM:

- pack B into cache-sized panels with deterministic column order,
- use a register tile such as 4x4 or 6x4,
- keep explicit `f64::mul_add` accumulation order per output lane,
- benchmark against the same worker before/after,
- prove panel packing preserves row/column/output order and exact accumulation sequence.
