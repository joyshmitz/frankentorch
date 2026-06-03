# ft-kernel-cpu scalar-unary parallel threshold

- Target: `unary_f64` scalar-map path (`exp/ln/sin/cos/tanh/gelu/silu/erf/sigmoid/...`)
- Lever: gate this path on a dedicated `SCALAR_UNARY_PARALLEL_THRESHOLD = 1<<19`
  (524288) instead of the shared `PARALLEL_THRESHOLD = 8192`
- File: `crates/ft-kernel-cpu/src/lib.rs`

## Profile Target

`tensor_exp` and the other transcendental unaries were 86–97x off the PyTorch
CPU anchors at small/medium sizes (`exp_10000` 13.4 us in torch vs ~1.3 ms
here). Decomposition isolated the cause:

- `relu/10000` (SIMD path) = 46 us, but `exp/10000` (scalar `value.exp()`) =
  ~1.3 ms — same tape/dispatch/record path, so per-op overhead is NOT the issue.
- The scalar map routed through `unary_f64`, which parallelised at `numel >=
  8192`. A same-worker A/B (RAYON_NUM_THREADS=1 vs default on one 64-core worker)
  showed the rayon split/join/collect overhead was a NET LOSS until ~0.5M:

```text
              serial (1 thread)   parallel (64 threads)
exp/10000     149 us              833 us     (5.6x slower parallel)
exp/100000    1.28 ms             2.63 ms    (2.0x slower parallel)
exp/1000000   19.5 ms             9.0 ms     (2.2x faster parallel)
```

So small/medium transcendentals were paying for parallelism that made them
slower. `8192` is right for cheap/SIMD/binary ops but far too low for this
per-element-libm path.

## Lever

One dedicated constant for the scalar-unary path: parallelise only at
`numel >= 524288`. Kept `<= 1M` so `exp/1000000` retains its 2.2x parallel win.
The cheap/SIMD unary ops and binary ops keep the original `8192` gate.

## Isomorphism Proof

- The scalar map is elementwise with NO cross-element accumulation, so serial
  and parallel produce bit-for-bit identical output regardless of scheduling —
  this lever only changes which loop runs, never a value.
- `cargo test -p ft-kernel-cpu --lib exp` → 89 passed, 0 failed.

## Result (default mode, patched binary)

```text
exp/10000    ~1312 us -> 101 us   (~10x)
exp/100000   ~3356 us -> 1.13 ms  (~3x)
exp/1000000  ~6138 us -> 4.34 ms
```

Realises the proven serial path for <=512k. Benefits every `unary_f64` consumer
(exp, ln, log2/10, sin/cos/tan, asin/acos/atan, sinh/cosh, tanh, gelu, silu,
elu, erf/erfc, softplus, mish, sigmoid, rsqrt, ...).

## Opportunity Score

Impact 4 x Confidence 5 / Effort 1 = 20.0. Bit-exact, one constant, broad
same-worker-proven win on the transcendental unary band.

## Next Primitive

The residual transcendental gap vs torch is scalar libm `value.exp()` (~15-20 ns)
vs torch's vectorised SIMD exp (SLEEF, few ns). A bit-exact SIMD exp would have
to reproduce libm's reduction bit-for-bit (hard); the tractable next swing is a
fused single-pass `softmax` kernel (max/exp-sum/normalise) to cut its 5 traced
intermediate ops and match the exp accumulation order exactly.
