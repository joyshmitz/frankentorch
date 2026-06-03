# ft-kernel-cpu softmax/log_softmax row-parallel threshold

- Target: `softmax_dim_tensor_contiguous_f64` + `log_softmax_dim_tensor_contiguous_f64`
  last-dim (`inner_size == 1`) row-parallel path
- Lever: gate the per-row `par_chunks` on `numel >= SOFTMAX_PARALLEL_NUMEL_THRESHOLD`
  (1<<16 = 65536); below it run the identical reduction serially
- File: `crates/ft-kernel-cpu/src/lib.rs`

## Profile Target

`softmax/vocab` ([32, n] softmax over n) was 80x off the PyTorch CPU anchor at
the common classifier/attention size (vocab/128: 556 us vs torch 6.98 us). The
per-element cost fell sharply with size (135 ns/elem at numel 4096 -> 4.3
ns/elem at 262144), i.e. a large FIXED overhead. Cause: the last-dim path
unconditionally spread the (few, tiny) rows across the 64-thread rayon pool;
rayon split/join dominated the trivial work.

Same-worker A/B (RAYON_NUM_THREADS=1 vs default, 64-core):

```text
numel    serial     parallel
4096     72 us      556 us    (7.7x slower parallel)
16384    214 us     676 us    (3.2x slower parallel)
65536    1.03 ms    1.02 ms   (break-even)
262144   2.74 ms    1.13 ms   (2.4x faster parallel)
```

## Lever

One constant: parallelise rows only at `numel >= 65536` (the measured
crossover); smaller softmaxes run the same reduction over `chunks_mut`. Applied
identically to softmax and log_softmax (the NLLLoss/cross-entropy hot path).

## Isomorphism Proof

- Each row's reduction (max -> exp -> pairwise_sum -> normalise) is independent
  and computed identically regardless of which thread runs it, so serial and
  parallel outputs are bit-for-bit identical — this only changes scheduling.
- `cargo test -p ft-kernel-cpu --lib softmax` -> 13 passed, 0 failed.

## Result (default mode, patched binary)

```text
softmax/vocab/128   556 us -> 52.7 us  (10.6x)
softmax/vocab/512   676 us -> 207.9 us (3.25x)
softmax/vocab/2048  1.02 ms -> 963 us  (parallel retained)
softmax/vocab/8192  1.13 ms -> 1.14 ms (parallel retained)
```

vocab/128 vs PyTorch: 80x -> 7.5x.

## Opportunity Score

Impact 4 x Confidence 5 / Effort 1 = 20.0. Bit-exact, one constant, 3-10x on the
most common softmax/log_softmax shapes; large-size parallel win preserved.

## Next Primitive

Residual softmax/exp gap is scalar libm `value.exp()` (~15-20 ns) vs torch's
vectorised SIMD exp (SLEEF). A bit-exact SIMD exp must reproduce libm's argument
reduction bit-for-bit — the open hard problem for the whole transcendental band.
