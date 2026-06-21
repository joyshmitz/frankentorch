# frankentorch-kgs4.151 — direct grouped masked flash SDPA (GQA) f64

Date: 2026-06-21
Agent: cc
Lever owner follow-up to: `frankentorch-kgs4.cod-b-masked-gqa-20260621`
(IvoryDeer diagnosed the loss and specified this exact next lever).

## Lever

`tensor_scaled_dot_product_attention_gqa` (no-grad f64, additive `[seq_q,seq_k]` or
`[B*h_q,seq_q,seq_k]` mask, contiguous q/k/v) used to materialise the `group`-repeated
K/V heads via `repeat_kv_heads` (unsqueeze → expand → **contiguous copy of `B*h_q*S*D`** →
reshape) and then run the dense masked flash kernel over all `B*h_q` heads.

New: `ft_kernel_cpu::sdpa_forward_masked_gqa_f64` folds the GQA broadcast into the index —
each Q head `hq` reads K/V head `hq / group` directly, no expansion copy — and parallelises
per `(batch, q_head)` **plus** an inner split over the independent `BR`-row blocks so all 64
cores are used (GQA has only `B*h_q = 16` heads, which alone left 48 cores idle). The `group`
Q heads sharing a K/V head run concurrently and reuse the same hot K/V head out of cache.

## Correctness (bit-exact)

- `ft-kernel-cpu` lib test `sdpa_masked_gqa_f64_matches_expand_then_masked_bit_exact`
  asserts `to_bits()` equality vs the expand-then-`sdpa_forward_masked_f64` path, both for a
  shared `[seq_q,seq_k]` mask and a per-Q-head `[B*h_q,seq_q,seq_k]` mask.
- `ft-api` lib test `sdpa_gqa_masked_fastpath_matches_expanded` asserts the routed result
  equals manual K/V expansion + plain masked SDPA (locks the batch/head/stride arithmetic).
- Example `sdpa_masked_headtohead` GQA lane matches PyTorch `enable_gqa=True`:
  checksum `-6.194718e1`, rel-diff `3.18e-14` (MATCH).

## Measurement (same-host: FT release binary + local PyTorch 2.12.1+cpu)

Shape: Q `[B=2,h_q=8,S=512,D=64]`, K/V `[B=2,h_kv=2,S=512,D=64]` (group 4), shared
`[512,512]` additive mask. 64-core host (shared with peer agents → some variance).

| Path | FT GQA | PyTorch GQA | Verdict |
| --- | ---: | ---: | --- |
| baseline (expand-then-flash), 8 torch threads | `33.7 ms` | `4.63 ms` | FT **7.29x slower** |
| this lever, 8 torch threads (example default) | `4.04–4.19 ms` | `4.53–4.83 ms` | FT **1.08–1.19x faster** |
| this lever, 32 torch threads (release-scorecard convention) | `4.0–5.7 ms` | `2.28–2.42 ms` | FT **1.8–2.5x slower** |

Internal speedup vs the old GQA path: **~6–8x** (33.7 ms → 4.0–5.7 ms), thread-count
independent (the FT timing is taken before the PyTorch subprocess runs).

## Verdict

**KEEP — internal ~6–8x win, PyTorch loss at the 32-thread scorecard convention (1.8–2.5x).**

PyTorch's GQA kernel scales with threads to ~2.3 ms (32t); FrankenTorch's flash kernel is
softmax-`exp`-bound (`B*h_q*S*S = 524288` scalar `libm::exp` calls per forward) and floors
near ~4 ms — beating that needs a vectorised `exp`, which changes rounding and is blocked by
the absolute-parity policy. So this does **not** cross the PyTorch line at 32 threads.

It is kept regardless because it is bit-exact and **strictly faster than the prior code at
every thread count** (removing the `B*h_q*S*D` expansion copy + the idle-core waste), turning
a 7.29x pathology into a near-parity path on a production LLM op (GQA = Llama-2/3, Mistral).
At the example's 8-thread default it is in fact a marginal PyTorch win; at the official 32t
convention it remains a loss, recorded honestly as `0W / 1L / 0N`.

## Win/loss/neutral vs PyTorch (32t convention): `0W / 1L / 0N`

## Gates

- `cargo test -p ft-kernel-cpu --release --lib`: 528 passed, 0 failed, 2 ignored.
- `cargo test -p ft-api --release --lib sdpa_gqa`: 5 passed, 0 failed.
- `cargo test -p ft-conformance --release`: green (parity preserved).
