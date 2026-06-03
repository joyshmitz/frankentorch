# ft-api grid_sample Row-Parallelization Attempt

- Bead: `frankentorch-kgs4.10`
- Parent umbrella: `frankentorch-kgs4`
- Skills: `/profiling-software-performance`, `/extreme-software-optimization`
- Crate: `ft-api`
- Target benchmark: `ops_bench/grid_sample/8x32x64x64_bilinear`
- Outcome: **rejected**, source lever reverted (bench + correctness test kept)

## Profile Target

`grid_sample_f64` (the `Self::grid_sample_f64` worker behind `tensor_grid_sample`)
builds its `[N, C, out_h, out_w]` output with a serial `for n, h, w { coord; for c
{ sample } }` loop. Per output element it computes a source coordinate
(unnormalize + padding mode) and, for bilinear, gathers 4 taps from `input` and
blends them. grid_sample is a real op (spatial transformers, optical-flow warping,
deformable sampling) and torch threads it.

Baseline (serial, RAYON_NUM_THREADS=1, worker vmi1227854):

```text
grid_sample/8x32x64x64_bilinear   time:   [20.083 ms 21.355 ms 23.304 ms]
```

## Lever Attempted

Write into a pre-sized buffer and distribute the `batch*channels*out_h` output
ROWS across Rayon via `par_chunks_mut(out_w)` (row r -> n=r/(C*out_h),
c=(r/out_h)%C, h=r%out_h). The grid coordinate depends only on (n,h,w); the row
split recomputes it per channel, which is deterministic and yields identical
ix/iy (hence identical samples), so the result is bit-for-bit identical.

## Result

Optimized (full threads, same worker):

```text
grid_sample/8x32x64x64_bilinear   time:   [15.036 ms 16.643 ms 18.429 ms]
```

Delta: `21.36 ms -> 16.64 ms` p50 = **1.28x**. Score ~1.0, well below the 2.0 bar.

### Why it does not pay off

grid_sample is **memory-bandwidth / latency bound**, not compute-bound. Each
output element issues 4 *scattered* gathers into `input` (the bilinear corners at
data-dependent positions), and the kernel writes an 8 MB output. Despite exposing
16384-way row parallelism (8*32*64 rows), the speedup capped at 1.28x — the
all-core scattered-read + write traffic saturates the memory subsystem. The
row-split also recomputes the source coordinate once per channel (channels-fold
redundancy), which adds work that partially offsets the parallel gain. A
no-redundancy batch-only split would only expose `batch`-way (8) parallelism and
hits the same bandwidth wall. This matches the earlier finding that pure
gather/scatter and reduction kernels do not clear 2x.

## Isomorphism Proof For The Rejected Draft

While applied, the draft was behavior-equivalent:

- Ordering: the (n,h,w,c) -> linear-index mapping was preserved (row r writes the
  same contiguous `out_w` block the serial push wrote).
- Floating point: identical coordinate-unnormalize / padding (Zeros/Border/
  Reflection) / nearest-round / 4-tap bilinear arithmetic in the same operand
  order; coordinate recomputation is deterministic so ix/iy are bit-identical.
- A `to_bits()` proof test
  (`grid_sample_matches_independent_reference_bit_exact`, kept) compares the
  output against an independent serial reference across
  {Bilinear/Zeros, Bilinear/Reflection, Nearest/Border} over a 9216-element case
  and passed while the draft was applied (and still passes for the serial form).

## Gates

- `rch exec -- cargo test -p ft-api --lib grid_sample` passed (existing + new
  reference test) while the draft was applied.
- `rch exec -- cargo bench -p ft-api --bench ops_bench -- grid_sample` baseline
  and after-run captured above; after-run failed the Score gate.
- Source lever reverted to the serial loop; the `grid_sample/8x32x64x64_bilinear`
  bench and the independent-reference correctness test are retained as infra.
