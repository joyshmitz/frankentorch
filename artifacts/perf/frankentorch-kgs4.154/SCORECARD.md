# frankentorch-kgs4.154 — intra-head block parallelism for unmasked f32 flash SDPA

Date: 2026-06-21
Agent: cc
Fourth application of the nested-BR-block trick (f32 mirror of kgs4.153).

## Lever

`ft_kernel_cpu::sdpa_forward_f32` (the f32 flash kernel behind the f32 no-grad/grad SDPA
fast paths — f32 is the dominant inference/serving dtype) parallelised only over
`num_bh = B*H` heads; `B*H=16` left 48 of 64 cores idle. Each head's independent `BR`-row
blocks are now also split across the pool, **guarded** by `num_bh < rayon::current_num_threads()`
so head-heavy inputs keep the cheaper serial inner loop (no regression). Per-block compute
factored into one shared closure (same shape as the f64 kernels in kgs4.152/153).

## Correctness (bit-exact)

Nested and serial paths produce identical FT output — both report the **same** relative diff
to PyTorch (`1.29e-9` non-causal, `1.05e-8` causal, well within f32 tolerance) — so the split
changes no reduction order. ft-kernel-cpu lib 530 passed / 0 failed; ft-conformance green
(includes f32 SDPA goldens).

## Measurement (same-host, 32 torch threads = release-scorecard convention)

Shape `[B*H=16, S=512, D=64]`, RAW `sdpa_forward_f32` forward vs PyTorch f32
`F.scaled_dot_product_attention`, `example sdpa_f32_inference_headtohead`.

| Lane | FT serial (before) | FT nested (after) | PyTorch f32 | win before → after |
| --- | ---: | ---: | ---: | --- |
| non-causal | `3.01 ms` (2.35x) | `1.95–2.01 ms` | `6.6–7.1 ms` | `2.35x` → `3.27–3.63x` faster |
| causal | `2.45 ms` (2.87x) | `1.77 ms` | `7.0–7.4 ms` | `2.87x` → `3.96–4.19x` faster |

Nested-block kernel speedup: ~1.5x non-causal, ~1.4x causal. **FT's f32 flash kernel beats
PyTorch's f32 CPU SDPA by 3.3–4.2x** at this shape.

## Caveat (honest scope)

This is a RAW-kernel comparison (FT excludes session/API overhead; PyTorch is the full op).
The no-grad f32 *session* path (`scaled_dot_product_attention`, requires_grad=false) currently
returns `DenseTensor(UnsupportedDType(F32))` — a pre-existing API gap filed as
**frankentorch-y5ubx** (the RAW kernel and the f32 *grad* path are fine). Once y5ubx is fixed,
this kernel win surfaces end-to-end for f32 serving.

## Win/loss/neutral vs PyTorch (32t): `2W / 0N` (kernel-level; widens an existing win)

## Gates

- `cargo test -p ft-kernel-cpu --release --lib`: 530 passed, 0 failed, 2 ignored.
- `cargo test -p ft-conformance --release`: green.
