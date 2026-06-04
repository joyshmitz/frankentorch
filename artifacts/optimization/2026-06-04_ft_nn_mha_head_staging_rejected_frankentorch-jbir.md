# ft-nn no-grad MHA head staging rejected - frankentorch-jbir

## Target

- Bead: `frankentorch-jbir`
- Crate: `ft-nn`
- Criterion target: `multihead_attention/forward_1x1024x128_h8_nograd`
- Selection: `br ready --json` returned no ready beads. Active in-progress perf surfaces were peer-owned in `ft-kernel-cpu` and `ft-optim`, so this fallback stayed on the shifted `ft-nn` no-grad MHA residual after `frankentorch-m9qp`.

## Baseline

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-nn --bench nn_bench -- multihead_attention/forward_1x1024x128_h8_nograd --warm-up-time 1 --measurement-time 10 --sample-size 20
time: [36.544 ms 38.097 ms 40.156 ms]
```

Profile-time evidence:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-nn --bench nn_bench -- multihead_attention/forward_1x1024x128_h8_nograd --profile-time 5
completed on ts1; Criterion did not emit function-level attribution.
```

Source decomposition for the focused benchmark showed the residual path in `MultiheadAttention::no_grad_f64_self_attention_fast_path`: Q/K/V projection, Q/K/V head staging, K transpose, score BMM, pairwise softmax, value BMM, head concat, and output projection.

## Lever Attempted

One `ft-nn`-only lever: fuse no-grad self-attention head staging by packing/scaling Q in one pass and packing K directly into the transposed layout. This removes one Q scaling pass and one K-head intermediate before transpose while leaving the existing BMM, pairwise softmax, and output projection kernels unchanged.

## Proof

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-nn mha_no_grad_self_attention_fast_path_matches_forward_qkv_bits -- --nocapture
test result: ok. 1 passed.
```

Golden verification:

```text
sha256sum artifacts/optimization/golden_outputs/ft_nn_mha_no_grad_self_attention_fast_path_frankentorch-m9qp.txt
4e7d4354d60cc304bcb7565b9d6ef01a8d646a896bdc416f3e3b98b37e5dc7c9
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
... ft_nn_mha_no_grad_self_attention_fast_path_frankentorch-m9qp.txt: OK
```

Isomorphism obligations satisfied while the candidate was applied:

- Ordering preserved: Q/K/V/out projection order, row-major head order, Q scale-before-score order, BMM order, softmax order, value BMM order, and concat order unchanged.
- Tie-breaking unchanged: no comparisons or tie rules are introduced.
- Floating-point preserved: each Q element is still multiplied by `self.scale` before score BMM; BMM and pairwise softmax kernels are unchanged.
- RNG unchanged: no RNG path is touched.
- Grad behavior preserved: accepted path remains no-grad only; grad-enabled and cross-attention cases still fall back.

## Result

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-nn --bench nn_bench -- multihead_attention/forward_1x1024x128_h8_nograd --warm-up-time 1 --measurement-time 10 --sample-size 20
time: [39.813 ms 41.335 ms 42.846 ms]
```

- p50: `38.097 ms -> 41.335 ms`
- Result: regression; source hunk reverted
- Score: impact `0` x confidence `4` / effort `1` = `0.0`

## Next Primitive

The read-only `/repeatedly-apply-skill` pass identified a different structural residual: the current path materializes both `scores` and `attn_weights`, each about 64 MiB for `1x1024x128_h8`. The next attack is an in-place pairwise softmax over the score buffer that exactly matches `softmax_dim_tensor_contiguous_f64` while preserving row-parallelism.
