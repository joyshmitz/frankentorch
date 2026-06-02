# ft-nn MHA Scale Tensor Reuse Pass 18

Bead: `frankentorch-8vuz`

Skills: `/profiling-software-performance`, `/extreme-software-optimization`

## Profile Target

Subsystem: `ft-nn::MultiheadAttention::forward_qkv`

Representative command:

```bash
rch exec -- cargo bench -p ft-nn --bench nn_bench -- multihead_attention/forward_8x64x128_h8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Scenario: batch `8`, sequence length `64`, embedding dim `128`, `8` heads, deterministic session/input fixture.

The profiled loop created one identical `[batch, seq_q, head_dim]` scale tensor per head before `tensor_mul(q_h, scale_t)`.

## Baseline

Worker: `vmi1153651`

Result:

```text
multihead_attention/forward_8x64x128_h8 time: [45.732 ms 54.898 ms 66.442 ms]
```

Criterion warning: the command requested 20 samples in 5s and collected 20 samples in an estimated 8.7933s.

## Lever

Create the `[batch_size, seq_len_q, head_dim]` scale tensor once before the head loop and reuse that same non-grad tensor for every head.

This removes `num_heads - 1` identical scale tensor allocations and leaf nodes per forward for self-attention while keeping the operand shape expected by `tensor_mul`.

## Isomorphism Proof

- Head iteration order is unchanged.
- Q/K/V projection order is unchanged.
- For every head, the old RHS tensor was filled with `self.scale` and shape `[batch_size, seq_len_q, head_dim]`; the new reused RHS tensor has the same shape and identical values.
- Each `tensor_mul(q_h, scale_t)` still performs the same per-element floating-point operation in the same element order.
- The scale tensor has `requires_grad = false`, so reusing the node cannot change gradient accumulation obligations.
- Attention score `bmm`, softmax, weighted-value `bmm`, head concatenation, and output projection are unchanged.
- No RNG, tie-breaking, sorting, or diagnostic branch is touched.

## After

Worker: `vmi1149989`

Result:

```text
multihead_attention/forward_8x64x128_h8 time: [13.755 ms 14.531 ms 15.258 ms]
```

Delta by p50: `54.898 ms -> 14.531 ms`, about `73.5%` faster (`3.78x`). Confidence is capped because the after run landed on a different worker, but the delta is far above observed worker-level noise.

## Behavior Evidence

Golden fixture:

```text
artifacts/optimization/golden_outputs/ft_nn_mha_pass18.txt
sha256 d3b78b1fbb9f1cbd157bd420c28f40d678f5e9b75ab27e4a30a23fc47ecc03b5
```

Focused test:

```bash
rch exec -- cargo test -p ft-nn mha_scale_reuse_golden_output_matches_fixture -- --nocapture
```

Validation:

```text
PASS rch exec -- cargo test -p ft-nn mha_scale_reuse_golden_output_matches_fixture -- --nocapture
PASS rch exec -- cargo check -p ft-nn --all-targets
PASS rch exec -- cargo clippy -p ft-nn --all-targets --no-deps -- -D warnings
PASS rch exec -- cargo fmt -p ft-nn --check
PASS sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
PASS git diff --check
NONZERO ubs crates/ft-nn/src/lib.rs crates/ft-nn/benches/nn_bench.rs crates/ft-nn/Cargo.toml
```

UBS residual inventory: exit `1`, 99 critical equality-comparison false positives, 6730 warnings, 462 info items across the scanned ft-nn files. UBS reported its own formatting, clippy, cargo check, tests-build, audit, and cargo-deny sections clean; no unsafe blocks found. The new bench direct-index warning from the first UBS pass was removed before final validation.

## Score

Impact `3` x confidence `2` / effort `1` = `6.0`

Status: keep and close.
