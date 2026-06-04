# ft-nn no-grad MHA self-attention fast path - frankentorch-m9qp

## Target

- Bead: `frankentorch-m9qp`
- Crate: `ft-nn`
- Criterion target: `multihead_attention/forward_1x1024x128_h8_nograd`
- Profile context: `br ready --json` was empty; active owned surfaces were `ft-kernel-cpu`, `ft-dispatch`, and `ft-optim`, so this pass stayed on the unowned ft-nn MHA residual after the prior no-grad Linear projection win.

## Baseline

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-nn --bench nn_bench -- multihead_attention/forward_1x1024x128_h8_nograd --warm-up-time 1 --measurement-time 10 --sample-size 20
time: [101.29 ms 105.56 ms 109.90 ms]
```

## Lever

One lever in `crates/ft-nn/src/lib.rs`: for `MultiheadAttention::forward` self-attention only, add a guarded no-grad f64 CPU fast path that:

1. Computes Q, K, V projections in the existing Q/K/V order with `ft_kernel_cpu::linear_tensor_f64`.
2. Packs heads into the same row-major layout as reshape `[B,S,H,D]`, permute `[0,2,1,3]`, reshape `[B*H,S,D]`.
3. Multiplies Q by `self.scale` before the score BMM, matching the old operation order.
4. Runs `bmm -> softmax_dim(dim=2) -> bmm` through existing safe-Rust kernels.
5. Restores `[B,S,E]` layout and applies the existing output projection primitive.

The tempting fused `sdpa_forward_f64` shortcut was rejected for this lever because it uses a sequential softmax sum internally, while the existing MHA graph routes through `softmax_dim_tensor_contiguous_f64` and its pairwise row sum. This pass preserves that floating-point order.

## Result

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-nn --bench nn_bench -- multihead_attention/forward_1x1024x128_h8_nograd --warm-up-time 1 --measurement-time 10 --sample-size 20
time: [38.309 ms 39.345 ms 40.534 ms]
```

- p50: `105.56 ms -> 39.345 ms`
- Speedup: `2.68x`
- Time reduction: `62.7%`
- Score: impact `5` x confidence `4` / effort `2` = `10.0`

## Isomorphism

- Ordering preserved: Q, K, V projections, Q scaling, score BMM, softmax, value BMM, head concat, and output projection remain in the same order.
- Tie-breaking unchanged: no ordering comparisons or top-k/tie logic are introduced.
- Floating-point preserved: the fast path scales Q before BMM and reuses the existing BMM and pairwise-softmax kernels instead of the tolerance-only fused SDPA shortcut.
- RNG unchanged: MHA forward uses no RNG; module initialization is untouched.
- Grad behavior preserved: grad-enabled and cross-attention paths still use `forward_qkv`; accepted fast-path output is a non-grad tensor, so no-grad backward rejection remains.

## Proof

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-nn mha -- --nocapture`: passed, including `mha_no_grad_self_attention_fast_path_matches_forward_qkv_bits`.
- Golden output: `ft_nn_mha_no_grad_self_attention_fast_path_frankentorch-m9qp.txt`
- SHA256: `4e7d4354d60cc304bcb7565b9d6ef01a8d646a896bdc416f3e3b98b37e5dc7c9`

Additional check/clippy/fmt/checksum evidence is recorded in the commit message for the final landed change.
