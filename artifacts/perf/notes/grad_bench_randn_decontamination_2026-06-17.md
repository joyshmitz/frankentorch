# grad_* bench RNG decontamination + honest backward reprofile (c55sy)

Date: 2026-06-17 · Agent: cc · Commit: 4ad23c9c (bench fix) · Bead: frankentorch-c55sy

## Problem

The `*/grad_*` benches in `crates/ft-api/benches/ops_bench.rs` built their inputs
with `tensor_randn(LARGE)` **inside** `b.iter()` (plus a fresh `Session::new()`),
unlike the `nograd_*` benches which hoist setup out. So every Criterion sample paid
for fresh Gaussian RNG over 2–8.4M elements. The
`artifacts/perf/frankentorch-next-reprofile-20260617` "grad" hotspots were therefore
dominated by RNG, **not** the backward kernels. The norm/batch_norm backward kernels
(`ft-kernel-cpu::batch_norm_backward_f64/f32` ~8330/8479) are already fully
rayon-parallel — there was no missing-rayon kernel lever to chase.

## Fix (4ad23c9c)

Generate each bench's random inputs **once** outside `b.iter()` (extract to `Vec<f64>`
via `tensor_values`), then rebuild the `requires_grad` leaves per-iter via
`tensor_variable(data.clone(), ..)` — a cheap memcpy that keeps a fresh tape while
moving RNG out of the measured region. Bench-only; no library change.

## Honest reprofile (after fix)

| bench                          | contaminated | corrected        |
|--------------------------------|--------------|------------------|
| layer_norm/grad_2048x1024      | 165 ms       | 65 ms            |
| rms_norm/grad_2048x1024        | 156 ms       | 85 ms            |
| group_norm/grad_32x256x28x28   | 494 ms       | 339 ms           |
| batch_norm/grad_1d_8192x1024   | 693 ms       | 262 ms isolated / ~500 ms in-batch |
| batch_norm/grad_train_32x256x28x28 | 513 ms   | 330 ms           |

## Caveats / next

- **Contention noise:** batch_norm/grad_1d measured 262 ms run-alone but ~500 ms in a
  5-bench back-to-back run (identical code) — rch workers vary ~1.4x and contention
  inflates. Do NOT treat these absolutes as a baseline; re-measure with an anchored
  single-process same-worker A/B before claiming any lever (see feedback_same_worker_ab).
- The corrected times still include the per-iter leaf-rebuild copy (e.g. 8.4M f64 ≈
  67 MB for batch_norm grad_1d), so they slightly overstate the pure backward.
- **Residual lever question (follow-up bead):** batch_norm/group_norm grad backward
  is still ~300–500 ms with an already-parallel kernel — the cost likely sits in the
  ft-api tape path (save_for_backward clones / forward stats recompute), not the
  kernel. Needs same-worker anchored profiling; ft-api/lib.rs is peer-hot.
