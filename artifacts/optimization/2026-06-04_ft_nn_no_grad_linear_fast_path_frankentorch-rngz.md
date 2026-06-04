# ft-nn no-grad Linear fast path for MHA - frankentorch-rngz

## Target

- Bead: `frankentorch-rngz`
- Criterion target: `multihead_attention/forward_1x1024x128_h8_nograd`
- Crate: `ft-nn`
- Worker: `ts1`

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-nn --bench nn_bench -- multihead_attention/forward_1x1024x128_h8_nograd --warm-up-time 1 --measurement-time 10 --sample-size 20
```

Result:

```text
time: [60.462 ms 60.750 ms 60.999 ms]
```

## Lever

Add an ft-nn-only guarded f64 `Linear::forward` no-grad fast path. For exact dense
`[*, in] @ [out, in]^T + bias` shapes it calls the existing safe-Rust
`ft_kernel_cpu::linear_tensor_f64` primitive and returns a non-grad tensor.

All grad-enabled, non-f64, empty, zero-batch, and shape-mismatch cases fall back
to the existing transpose, matmul, reshape, expand, and add graph.

## Re-benchmark

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-nn --bench nn_bench -- multihead_attention/forward_1x1024x128_h8_nograd --warm-up-time 1 --measurement-time 10 --sample-size 20
```

Result:

```text
time: [54.614 ms 54.945 ms 55.415 ms]
```

Delta:

- p50 speedup: `1.106x`
- p50 time reduction: `9.6%`

## Isomorphism

- Grad-enabled `Linear::forward` continues through the existing graph path.
- Parameter ordering and `MultiheadAttention` Q/K/V/out projection ordering are unchanged.
- Accepted no-grad f64 linear cases produce the same tensor values as the transpose/matmul/bias path.
- Bias addition order, RNG behavior, tie behavior, and no-grad backward rejection are unchanged.
- Existing gradient golden fixture still passes.

Golden output:

```text
4e7d4354d60cc304bcb7565b9d6ef01a8d646a896bdc416f3e3b98b37e5dc7c9  artifacts/optimization/golden_outputs/ft_nn_mha_no_grad_linear_fast_path_frankentorch-rngz.txt
```

## Proof

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-nn mha -- --nocapture
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-nn linear -- --nocapture
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo check -p ft-nn --all-targets
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo clippy -p ft-nn --all-targets --no-deps -- -D warnings
cargo fmt -p ft-nn --check
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
```

Score: impact 2 x confidence 3 / effort 2 = 3.0.
