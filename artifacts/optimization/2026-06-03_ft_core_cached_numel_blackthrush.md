# Pass: ft-core cached TensorMeta numel

- Agent: BlackThrush
- Crate: `ft-core`
- Target benchmark: `tensor_meta/numel_rank8_repeated_65536`
- Bead: pending, `.beads/issues.jsonl` reserved by SilentSnow during validation

## Profile Target

No ready unclaimed perf bead was available. The fallback profile target was the
existing ft-core Criterion benchmark for repeated `TensorMeta::numel()` calls on
a rank-8 shape. The hot path recomputed the saturated product on every call even
though `TensorMeta` owns immutable shape metadata after construction.

Baseline via rch Criterion:

```text
worker: vmi1227854
command: env CARGO_TARGET_DIR=/data/tmp/frankentorch-blackthrush-ftcore-numel-baseline rch exec -- cargo bench -p ft-core --bench core_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
time: [237.91 us 244.75 us 251.84 us]
```

## One Lever

Store a saturated `numel` field in `TensorMeta` at construction time and return
that field from `TensorMeta::numel()`.

The cached value uses the same contract required by existing tests:

- scalar shape returns `1`
- any zero dimension returns `0`
- otherwise checked multiplication saturates to `usize::MAX` on overflow

## Isomorphism Proof

- Ordering: no collection order changes; shape, strides, dtype, device, and
  quantization remain constructed through the same APIs.
- Tie-breaking: no comparisons or tie-breakers are introduced.
- Floating point: the changed path performs no floating-point arithmetic.
- RNG: no RNG state is read or written.
- Error behavior: stride/rank/storage validation still runs after constructor
  metadata assembly and uses the same validation code.
- Golden output: existing ft-core numel fixture is unchanged and verified.

```text
sha256: fc134f8a2fbb18b29efcf4f4d8c09d3e78c8a0b4375fc586fb9035a658bff864
file: artifacts/optimization/golden_outputs/ft_core_numel_pass22.txt
```

## Result

After via rch Criterion:

```text
worker: vmi1153651
command: env CARGO_TARGET_DIR=/data/tmp/frankentorch-blackthrush-ftcore-numel-after rch exec -- cargo bench -p ft-core --bench core_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
time: [29.622 us 30.853 us 32.362 us]
```

Delta:

- p50: `244.75 us -> 30.853 us`
- improvement: about 7.9x faster
- confidence: cross-worker comparison, but effect size is far beyond worker
  noise and the target is a pure field-read replacement
- score: impact 5 x confidence 3 / effort 1 = 15.0
- decision: keep

## Gates

- `rch exec -- cargo test -p ft-core numel -- --nocapture`: pass after fixing
  zero-dimension-before-overflow semantics
- `rch exec -- cargo bench -p ft-core --bench core_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20`: pass before and after
- `sha256sum artifacts/optimization/golden_outputs/ft_core_numel_pass22.txt`:
  `fc134f8a2fbb18b29efcf4f4d8c09d3e78c8a0b4375fc586fb9035a658bff864`
