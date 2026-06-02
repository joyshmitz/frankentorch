# ft-kernel-cpu Softmax/LogSoftmax Parallel Pass

- Bead: `frankentorch-wcoo`
- Parent umbrella: `frankentorch-kgs4`
- Skills: `/extreme-software-optimization`, `/alien-graveyard`, `/alien-artifact-coding`
- Crate: `ft-kernel-cpu`
- Target benchmarks:
  - `log_softmax_f64_8192x1024_dim1`
  - `log_softmax_f32_8192x1024_dim1`
  - `softmax_f64_strided_4096x32x8_dim1`

## Profile Target

The residual torch/ATen gap was the serial slice loop in f32 softmax, both
log_softmax fast paths, and the general strided softmax/log_softmax paths.
Each reduced slice is independent and exp-heavy, so this is a direct
parallel-morsel target.

Baseline:

```text
worker: vmi1153651
command: rch exec -- env RAYON_NUM_THREADS=1 cargo bench -p ft-kernel-cpu --bench softmax_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
log_softmax_f64_8192x1024_dim1: [186.14 ms 194.54 ms 203.67 ms]
log_softmax_f32_8192x1024_dim1: [103.67 ms 110.23 ms 117.19 ms]
softmax_f64_strided_4096x32x8_dim1: [20.372 ms 21.463 ms 22.571 ms]
```

After:

```text
worker: vmi1153651
command: rch exec -- cargo bench -p ft-kernel-cpu --bench softmax_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
log_softmax_f64_8192x1024_dim1: [44.133 ms 49.905 ms 57.147 ms]
log_softmax_f32_8192x1024_dim1: [24.366 ms 28.905 ms 35.231 ms]
softmax_f64_strided_4096x32x8_dim1: [7.1350 ms 8.9832 ms 11.404 ms]
```

Delta:

- f64 log_softmax p50: `194.54 ms -> 49.905 ms`, about 3.90x faster.
- f32 log_softmax p50: `110.23 ms -> 28.905 ms`, about 3.81x faster.
- f64 strided softmax p50: `21.463 ms -> 8.9832 ms`, about 2.39x faster.
- Score: Impact 4 x Confidence 4 / Effort 2 = 8.0.

## Alien Recommendation Card

Change: parallelize independent softmax/log_softmax slices with Rayon
`par_chunks_mut`, using chunk boundaries that match output slice ownership.

Mapped graveyard sections:

- High-level summary: vectorized execution plus morsel-driven parallelism
  processes data in cache-sized batches and aims for near-linear scaling where
  chunks are independent.
- Alien graveyard appendix: every optimization must name a baseline comparator
  and record the benchmark harness.
- Risk matrix: constants can dominate, so keep the pre-existing serial/small
  softmax fast path where it already exists and accept only measured wins.

Expected value: Impact 4 * Confidence 4 * Reuse 4 / Effort 2 /
AdoptionFriction 1 = 32.0.

Fallback: revert only this softmax/log_softmax parallel chunking lever if any
bit-exact test, golden checksum, or Criterion score regresses. No public API,
error class, dtype, or layout rule changes.

## Alien Artifact Proof

Selected family: certified independent-slice parallelization.

Proof obligations:

- Ordering: output indices remain identical because each Rayon chunk is a
  disjoint contiguous output block and each block writes the same local offsets
  as the serial loop.
- Tie-breaking: no arg/tie comparisons are introduced.
- Floating point: each reduced slice preserves the same gather order, max fold,
  exp calls, pairwise sum, and scatter order. Parallelism occurs only between
  independent slices, not inside a slice.
- RNG: softmax/log_softmax use no RNG.
- Errors: dtype/layout/dimension/storage validation remains before the parallel
  branch.
- Golden output: selected f64/f32 fast-path and strided-path bit patterns are
  pinned by sha256
  `f826f2307c88a854d3586ac640ab233b7d8ae10bb7731e84f062e4dadaa242ab`.

## Gates

- `rch exec -- env RAYON_NUM_THREADS=1 cargo bench -p ft-kernel-cpu --bench softmax_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20` passed.
- `rch exec -- cargo bench -p ft-kernel-cpu --bench softmax_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20` passed.
- `rch exec -- cargo test -p ft-kernel-cpu softmax_family_parallel -- --nocapture` passed: 2 tests.
- `rch exec -- cargo check -p ft-kernel-cpu --all-targets` passed.
- `rch exec -- cargo clippy -p ft-kernel-cpu --all-targets --no-deps -- -D warnings` passed.
- `rch exec -- cargo fmt -p ft-kernel-cpu --check` passed.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed.
- `git diff --check` passed.
