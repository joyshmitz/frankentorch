# frankentorch-m4pro-whisper-sgemm sgemm Encoder-Overhead Rejection Evidence

## Target

- Crate: `ft-kernel-cpu`
- Host: Apple M4 Pro (14 cores: 10 P + 4 E), arm64, macOS — the consumer
  (`franken_whisper`) production host.
- Consumer profile: the whisper-large encoder window spends ~4.4 s in
  `matmul_tensor_contiguous_f32` across three GEMM shapes repeated over 32
  layers:
  - `[1500,1280] x [1280,1280]` — attention q/k/v/out projections
  - `[1500,1280] x [1280,5120]` — MLP up-projection
  - `[1500,5120] x [5120,1280]` — MLP down-projection
- All three are TALL (m = 1500 ≫ the column gate's `n > 4*m` ratio is never
  met), so they take the **row-split** parallel path, not the column path.
- Benchmark: `cargo bench -p ft-kernel-cpu --bench gemm_bench --
  matmul_whisper_f32` (added in `4af78e91`).

## Attempted levers

Three candidate levers were proposed against the encoder's sgemm overhead.

### Lever 1 — `sgemm_col_parallel` per-block `ct` Vec reuse — NOT APPLICABLE

The per-call `vec![0.0; m*bw]` block buffer in `sgemm_col_parallel` only
exists on the **column** path. Every whisper-large encoder shape is TALL and
takes the **row-split** path (`should_parallelize_cols` requires `n > 4*m`;
here `n` ∈ {1280, 5120} vs `4*m = 6000`, never satisfied). The `ct` buffer is
never allocated for these shapes, so there is nothing to reuse. Lever rejected
as inapplicable to the consumer's hot shapes (no code changed for it).

### Lever 3 — row-split block-size sweep on M4 Pro — REGRESSES

Hypothesis: 1 block/thread hands an equal row band to each of the 4 slow E
cores, which then straggle the join; splitting into a few blocks PER thread
would let rayon work-stealing rebalance onto the P cores.

`block_rows` was changed to `m.div_ceil(threads * F)` and swept on M4 Pro
(`--baseline ft-pre`, warm-up 2 s, measure 8 s, 20 samples):

| shape (m×k×n)      | F=1 (baseline) | F=2          | F=3          |
| ------------------ | -------------- | ------------ | ------------ |
| 1500×1280×1280     | 8.41 ms        | +2.3% (n.s.) | +1.8% (n.s.) |
| 1500×1280×5120 fc  | 35.98 ms       | **+8.7%**    | **+16.6%**   |
| 1500×5120×1280 prj | 35.41 ms       | +2.2% (n.s.) | **+8.9%**    |

Oversubscription REGRESSES the MLP shapes monotonically: smaller row bands
re-stream the (cache-overflowing, 26 MB) B panel more times, and that B-traffic
cost dwarfs any straggler savings. The existing **1 block / thread** is optimal
for these shapes. Lever reverted; `block_rows` is unchanged.

### Lever 2 — output-buffer reuse (`matmul_tensor_contiguous_f32_into`) — NEUTRAL

`matmul_tensor_contiguous_f32` allocs and zero-inits a fresh `vec![0.0; m*n]`
each call (up to 30 MB for the MLP up-projection). An additive, buffer-reusing
`matmul_tensor_contiguous_f32_into` entry point was added (committed in
`4af78e91`) and benched with a warm, pre-sized scratch (the way a consumer
encoder reuses one buffer per shape across all 32 layers).

Same-host A/B, alloc path vs `_into` warm-reuse path (M4 Pro, two readings):

| shape              | alloc path        | `_into` warm reuse | delta        |
| ------------------ | ----------------- | ------------------ | ------------ |
| 1500×1280×1280     | 8.41 / 9.34 ms    | 8.81 / 9.57 ms     | within noise |
| 1500×1280×5120 fc  | 35.98 / 32.66 ms  | 34.87 / 35.22 ms   | within noise |
| 1500×5120×1280 prj | 35.41 / 33.40 ms  | 34.87 ms           | within noise |

The alloc+zero-init is a sub-millisecond fraction of a compute-bound GEMM
(8–36 ms), and on macOS the `vec![0.0;…]` is calloc-lazy (pages are zeroed on
first touch by the GEMM regardless), so removing it buys nothing measurable.
The delta is below the host's run-to-run variance on every shape.

## Consumer e2e attempt

The consumer (`franken_whisper`) encoder was wired through `_into` via a
per-window reusable scratch (`EncoderScratch`: q/k/v/proj/mlp_h + an LN-input
buffer reused across all 32 layers, routed through a new `nn::matmul_bias_into`).

- **Bit-exactness gate: PASSED.** The optimized `native_ab` binary produced
  byte-identical golden JSON to the pre-change REF binary on BOTH models
  (`tiny.en` sha `b4c69e84…`, `large-v3-turbo` sha `f7ce8a98…`; both match the
  `/tmp/fw_golden` goldens' transcript text + every segment timestamp). The
  GEMM runs with `beta = 0` and overwrites every output cell, so buffer reuse
  cannot leak prior contents — confirmed empirically.
- **e2e wall A/B: INCONCLUSIVE → no demonstrable win.** Interleaved REF-vs-OPT
  wall A/B (jfk, large) could not produce a clean signal: the measurement host
  was at load average ~24 on 14 cores, with single-run encoder samples spanning
  6.9–9.9 s (±43% within one criterion run) and pair-to-pair sign flips. No
  robust min/p25 win emerged across pairs; the criterion `encoder_window_large`
  CI was [+6.5%, +56%] — pure load noise, dominated by contention, not a real
  effect. Under house discipline (interleaved, min/p25, ≥6 stable pairs) the
  lever does NOT clear the bar.

## Proof / gates

- `cargo test -p ft-kernel-cpu`: **404 passed, 0 failed** (incl.
  `gemm_col_split_matches_single_bit_exact`,
  `gemm_row_split_matches_single_bit_exact`, and the 5
  `matmul_tensor_contiguous_*` tests covering the `_into`-backed wrapper).
- `cargo test -p ft-dispatch`: **108 passed, 0 failed.**
- Bit-exactness: `matmul_tensor_contiguous_f32` (now a thin wrapper over
  `_into`) is byte-identical to the prior fresh-alloc implementation —
  k-accumulation order is unchanged and `beta = 0` overwrites every cell.

## Decision

- **Rejected as a measurable optimization** for the whisper-large encoder on
  M4 Pro. Lever 1 is inapplicable (wrong path), Lever 3 regresses (B re-stream
  cost), Lever 2 is neutral (compute-bound; calloc-lazy alloc is free).
- The additive, bit-exact `matmul_tensor_contiguous_f32_into` API + the
  `matmul_whisper_f32{,_into}` benches were retained (committed `4af78e91`) as
  permanent measurement infrastructure and a zero-cost convenience entry point;
  they introduce no behavior change and the row-split `block_rows` heuristic is
  unchanged. The consumer-side `EncoderScratch` wiring was reverted — it added
  threading churn for no e2e win.
- Root cause (consistent with the consumer's own pass-3 finding): the encoder
  is GEMM (M = 1500, weights reused across rows) and **compute-bound**. Per-call
  allocation and block-straggler effects are both second-order to the
  `matrixmultiply` microkernel's arithmetic + B-panel cache traffic. There is no
  free win here without changing the microkernel itself (see the rejected
  packed-panel/Strassen pilots, 20260603T18*).
- Score: impact 0 (neutral) × confidence 4 / effort 2 = 0.0 → below the ≥ 2.0
  implement gate.
