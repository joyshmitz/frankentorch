# ft-nn MHA in-place score softmax (frankentorch-ob2q)

## Target

- Skill loop: `/repeatedly-apply-skill` applying `/extreme-software-optimization`, with `/alien-graveyard` and `/alien-artifact-coding` discipline for structural pivots.
- Bead: `frankentorch-ob2q`
- Crate: `ft-nn`
- Benchmark: `multihead_attention/forward_1x1024x128_h8_nograd`
- Profile-backed source target: `MultiheadAttention::no_grad_f64_self_attention_fast_path` materialized both the attention score matrix and a second softmaxed attention-weight matrix before the value BMM.

The no-grad self-attention fast path already computes:

```text
scores = bmm(q_heads, k_t)
attn_weights = softmax(scores, dim=2)
head_out = bmm(attn_weights, v_heads)
```

For `1x1024x128_h8`, `scores` and `attn_weights` are each 8 x 1024 x 1024 f64 elements, so the second matrix costs about 64 MiB of allocation and memory traffic.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-nn --bench nn_bench -- multihead_attention/forward_1x1024x128_h8_nograd --warm-up-time 1 --measurement-time 10 --sample-size 20
```

Worker/result:

```text
ts1: [69.597 ms 71.714 ms 73.797 ms]
```

Profile-time command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-nn --bench nn_bench -- multihead_attention/forward_1x1024x128_h8_nograd --profile-time 10
```

Result: completed on `ts1`; Criterion did not emit function attribution, so the source-level target is the explicit score-softmax-value chain above.

## Lever

Replace the no-grad fast path's call to `ft_kernel_cpu::softmax_dim_tensor_contiguous_f64` with an exact in-place last-dim f64 softmax over the `scores` buffer, then feed `scores` directly into the existing value BMM.

This is one lever: remove the second attention-weight allocation while preserving the existing BMM kernels and the same fast-path shape guard.

## Isomorphism proof

- Scope is unchanged: only the no-grad f64 CPU self-attention fast path is touched. Grad-enabled, cross-attention, non-f64, non-CPU, zero-sized, overflow, and shape-mismatch cases still use the existing fallback paths.
- Q, K, V, and output projection order is unchanged.
- Head packing order, Q scaling, score BMM order, value BMM order, and concatenation order are unchanged.
- The softmax row algorithm matches the replaced kernel path: row max fold, exp write order, `pairwise_sum_f64` recursion with block size 128, divide order, and row-parallel threshold 65536.
- Floating-point behavior is preserved for the accepted path because the same row-local operations occur in the same order. NaN and Inf propagation follows the same comparisons, exponentiation, summation, and division.
- Tie-breaking and RNG behavior are not involved in this path.
- Golden fixture checksums are unchanged:
  - `ft_nn_mha_no_grad_self_attention_fast_path_frankentorch-m9qp.txt`: `4e7d4354d60cc304bcb7565b9d6ef01a8d646a896bdc416f3e3b98b37e5dc7c9`
  - `ft_nn_mha_no_grad_linear_fast_path_frankentorch-rngz.txt`: `4e7d4354d60cc304bcb7565b9d6ef01a8d646a896bdc416f3e3b98b37e5dc7c9`

## Validation

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-nn mha_no_grad -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-nn --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-nn --all-targets --no-deps -- -D warnings
cargo fmt -p ft-nn -- --check
sha256sum artifacts/optimization/golden_outputs/ft_nn_mha_no_grad_self_attention_fast_path_frankentorch-m9qp.txt artifacts/optimization/golden_outputs/ft_nn_mha_no_grad_linear_fast_path_frankentorch-rngz.txt
```

Result: pass. The remote check/clippy output includes unrelated dependency warnings from peer-owned active surfaces; `ft-nn` passed. The local format check was used for the non-compilation fmt command. After replacing the recursive checksum-equivalent slices with `split_at(mid)` to avoid a new UBS-visible direct-slicing pattern, the same focused test/check/clippy/fmt proof was repeated successfully on the final source.

UBS note: `ubs crates/ft-nn/src/lib.rs crates/ft-nn/Cargo.toml` exits 1 on a broad pre-existing `ft-nn/src/lib.rs` inventory, including false-positive "secret comparison" reports on ordinary shape/name comparisons. The new hunk avoids wildcard imports and adds no unsafe code or unwraps.

## Re-benchmark

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-nn --bench nn_bench -- multihead_attention/forward_1x1024x128_h8_nograd --warm-up-time 1 --measurement-time 10 --sample-size 20
```

Worker/result:

```text
ts1: [28.200 ms 28.759 ms 29.729 ms]
```

Delta:

```text
p50 71.714 ms -> 28.759 ms
speedup 2.49x
time reduction 59.9%
```

## Score

```text
Impact 5 x Confidence 4 / Effort 2 = 10.0
```

Kept: the change clears the `>=2.0` score threshold with same-worker RCH evidence and exact behavior proof.
