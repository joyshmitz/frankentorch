# ft-nn Fused QKV Projection Rejection

- Bead: `frankentorch-ft-nn-fused-qkv-projection-ua44`
- Target benchmark: `multihead_attention/forward_8x64x128_h8`
- Lever attempted: for self-attention only, concatenate Q/K/V projection weights
  and biases, run one wider projection matmul, then narrow Q/K/V chunks.

## Baseline

Fresh current-tree rch Criterion on `ts2`:

```text
cargo bench -p ft-nn --bench nn_bench -- multihead_attention/forward_8x64x128_h8 --warm-up-time 1 --measurement-time 5 --sample-size 20
time: [31.033 ms 31.281 ms 31.713 ms]
```

Existing artifacts identify `MultiheadAttention::forward_qkv` as the residual
MHA hotspot after the batched-head and self-flat-reuse passes.

## Alien Primitive Mapping

- Graveyard primitive: graph batching / operator-fusion. The attempted rewrite
  batches three independent projection matmuls into one wider projection.
- Alien artifact proof target: certified numerical-graph rewrite with exact
  forward values and exact autograd evidence for all original parameters.

## Isomorphism Result

- Forward ordering: matched the existing golden output values exactly.
- Tie-breaking: unchanged; MHA forward adds no comparison tie-breaker.
- RNG: unchanged; the attempted forward path added no random draw.
- Floating point: forward per-output dot order was preserved, but backward input
  accumulation order changed at the bit level.
- Autograd: rejected. The golden fixture detected input-gradient drift in the
  last bits. Example drift:
  - expected `x_grad[0] = 0.24736662142461532`
  - got `x_grad[0] = 0.24736662142461538`

Golden fixture used:

```text
artifacts/optimization/golden_outputs/ft_nn_mha_self_flat_reuse_frankentorch-l3mm.txt
sha256: daffba8149e995d8fb07284e5edcbd18f5593fbf8fd3c6852604faa1f0cabb15
```

## Validation

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-nn mha_self_flat_reuse_golden_output_matches_fixture -- --nocapture
```

Result: failed on `ts2` because the fixture's output/loss/parameter gradients
matched, but `x_grad` differed by last-bit rounding from the changed backward
accumulation order.

## Decision

Rejected. Score: Impact 0 x Confidence 5 / Effort 1 = 0.0 because behavior
parity failed before benchmarking the candidate.

No source change was kept. The `crates/ft-nn/src/lib.rs` hunk was manually
reverted.

## Next Primitive

Do not retry this as a graph-only fused projection. The next MHA attack needs a
fundamentally different primitive with an explicit backward-order certificate,
such as a safe-Rust fused attention primitive whose backward accumulator emits
input gradients in the same Q-then-K-then-V contribution order as the existing
graph, or a profiler-backed memory-layout rewrite below the autograd graph that
keeps the current node topology intact.
