# frankentorch-kgs4.152 — intra-head block parallelism for masked f64 flash SDPA

Date: 2026-06-21
Agent: cc
Builds on kgs4.151 (same nested-BR-block trick, now applied to the dense kernel).

## Lever

`ft_kernel_cpu::sdpa_forward_masked_f64` (the masked f64 flash kernel behind the
non-GQA `scaled_dot_product_attention` masked primary/tensor lanes, and the GQA-grad
path) parallelised only over `num_bh = B*H` heads. For the standard transformer shape
`B*H = 16` on a 64-core host that leaves 48 cores idle. Each head's `BR`-row blocks are
fully independent, so the kernel now also splits them across the pool — but **only when
`num_bh < rayon::current_num_threads()`** (head-heavy inputs already saturate the outer
split, so they keep the cheaper serial inner loop with one reused scores buffer: no
regression, no extra task overhead). The per-block compute is factored into one closure
shared by both paths.

## Correctness (bit-exact)

Output checksum `-9.829413e0` is byte-identical to the pre-change kernel; example
`sdpa_masked_headtohead` rel-diff vs torch unchanged at `3.36e-14` (MATCH). Per-`BR`-block
work is independent (own scores, own output rows, shared read-only K/V/mask), so the split
changes no reduction order. ft-kernel-cpu sdpa tests pass; ft-conformance green (199 lib +
all bins).

## Measurement (same-host, 32 torch threads = release-scorecard convention)

Shape `[B*H=16, S=512, D=64]`, shared `[512,512]` additive mask, no-grad f64.

| Lane | FT before | FT after | PyTorch (32t) | win before → after |
| --- | ---: | ---: | ---: | --- |
| primary masked | `7.53 ms` | `6.14–6.26 ms` | `~14.0 ms` | `1.85x` → `2.21–2.33x` faster |
| tensor masked | `7.13 ms` | `6.36 ms` | `~13.9 ms` | `1.95x` → `2.23–2.37x` faster |

~1.2x kernel speedup on top of an already-winning lane; the ceiling is the softmax-`exp`
cost (scalar `libm::exp`), which the extra cores cannot remove (see kgs4.151).

## Win/loss/neutral vs PyTorch (32t): `1W / 0N` (widens an existing win; no regression)

## Gates

- `cargo test -p ft-kernel-cpu --release --lib sdpa`: pass.
- `cargo test -p ft-conformance --release`: 199 lib + bins/integration/smoke/doctests green.
