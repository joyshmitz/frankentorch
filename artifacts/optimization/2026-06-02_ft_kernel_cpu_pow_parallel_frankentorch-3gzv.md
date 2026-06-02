# ft-kernel-cpu Parallel Pow Pass

- Bead: `frankentorch-3gzv`
- Parent umbrella: `frankentorch-kgs4`
- Skills: `/extreme-software-optimization`, `/alien-graveyard`, `/alien-artifact-coding`
- Crate: `ft-kernel-cpu`
- Target benchmark: `pow_f64_1m_exp2.5`, `pow_f32_1m_exp2.5`

## Profile Target

Large contiguous `pow` maps are compute-bound and remain a direct torch/ATen
elementwise throughput gap. The benchmark holds the input layout and function
constant while toggling Rayon thread count on the same `rch` worker.

Baseline:

```text
worker: vmi1153651
command: rch exec -- env RAYON_NUM_THREADS=1 cargo bench -p ft-kernel-cpu --bench elementwise_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
pow_f64_1m_exp2.5: [28.477 ms 29.351 ms 30.558 ms]
pow_f32_1m_exp2.5: [15.284 ms 15.712 ms 16.293 ms]
```

After:

```text
worker: vmi1153651
command: rch exec -- cargo bench -p ft-kernel-cpu --bench elementwise_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
pow_f64_1m_exp2.5: [10.868 ms 11.541 ms 12.524 ms]
pow_f32_1m_exp2.5: [4.8638 ms 5.3981 ms 6.1632 ms]
```

Delta:

- f64 p50: `29.351 ms -> 11.541 ms`, about 2.54x faster.
- f32 p50: `15.712 ms -> 5.3981 ms`, about 2.91x faster.
- Score: Impact 3 x Confidence 3 / Effort 1 = 9.0.

## Alien Recommendation Card

Change: parallelize the large contiguous f64/f32 pow map through Rayon while
retaining the existing serial path below `PARALLEL_THRESHOLD`.

Mapped graveyard sections:

- `alien_cs_graveyard.md` numeric-kernel guidance: focus on cache locality,
  SIMD/vectorized maps, and benchmarked constants.
- `alien_cs_graveyard.md` appendix: accept only with a baseline comparator,
  environment, and golden artifact.
- High-level summary vectorized-execution primitive: process independent data
  morsels in bulk when the operation has no cross-element dependency.

Expected value: Impact 3 * Confidence 3 * Reuse 3 / Effort 1 /
AdoptionFriction 1 = 27.0.

Fallback: keep the serial iterator path for small tensors, and manually revert
the parallel branch if the bit-exact test, golden checksum, or Criterion score
fails.

## Alien Artifact Proof

Selected family: certified pure-map parallelization.

Proof obligations:

- Ordering: output indices remain exactly the input index order because Rayon
  `collect::<Vec<_>>()` preserves indexed parallel iterator order.
- Tie-breaking: no comparisons or tie-breakers are introduced.
- Floating point: each output element calls the same `powf_torch_signed_zero_*`
  helper once; there is no reduction, reassociation, or cross-element state.
- RNG: pow uses no RNG.
- Errors: layout/storage validation remains before the parallel branch.
- Threshold: tensors below `PARALLEL_THRESHOLD` keep the existing serial path.
- Golden output: selected f64/f32 output bit patterns over a 16K-element tensor
  exercise the parallel branch and are pinned by sha256
  `d37527a964e75074876f3d3e89322dad24bede9570ec0a1ede68a2bbba1a4ced`.

## Gates

- `rch exec -- env RAYON_NUM_THREADS=1 cargo bench -p ft-kernel-cpu --bench elementwise_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20` passed.
- `rch exec -- cargo bench -p ft-kernel-cpu --bench elementwise_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20` passed.
