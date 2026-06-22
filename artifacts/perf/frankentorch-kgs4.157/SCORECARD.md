# frankentorch-kgs4.157 - PReLU borrowed-input + deterministic channel-parallel backward

Assignee: cod-a / IvoryDeer
Date: 2026-06-21

## Lever

Targeted the f64 PReLU train-step gap against PyTorch. The old autograd path saved
owned copies of the full input and weight tensors into `FunctionCtx`, then ran a
serial backward loop. The kept candidate:

- switches PReLU to `tensor_apply_function_f64_borrowed_inputs`, so backward reads
  immutable input/weight slices from the tape instead of cloning them into context;
- computes `grad_x` with a parallel indexed map for large tensors;
- computes `grad_w` independently per channel, preserving the same per-channel
  accumulation order as the old serial loop for deterministic gradients.

This is the alien-graveyard/polyhedral lever for this pass: split the affine
NCH iteration space into independent channel reductions and remove dead staging
copies at the autograd boundary.

## Measurement

Harness: `crates/ft-api/examples/prelu_train_borrowed_h2h.rs`

Shape: f64 PReLU train step `[32,512,256]`, scalar loss
`prelu(x, weight).sum().backward()`, PyTorch CPU `2.12.1+cpu`, checksum relative
error `1.966e-14`.

RCH note: two `rch exec -- cargo run --release -p ft-api --example
prelu_train_borrowed_h2h` attempts selected cold `ovh-a` despite the warm
`CARGO_TARGET_DIR`; both were interrupted before completion to avoid a cold
remote target build. Measurements used the existing local warm target with
`cargo +nightly-2026-06-09`, matching the target's rustc metadata.

| Variant | FT median | PyTorch median | Ratio vs PyTorch | FT delta |
| --- | ---: | ---: | ---: | ---: |
| baseline saved-context serial, 8 iters | 145.510 ms | 34.624 ms | FT 4.20x slower | 1.00x |
| borrowed-input only, 8 iters | 121.958 ms | 37.450 ms | FT 3.26x slower | 1.19x faster |
| borrowed + channel-parallel, 8 iters | 103.712 ms | 36.752 ms | FT 2.82x slower | 1.40x faster |
| borrowed + channel-parallel, 20 iters | 93.263 ms | 36.136 ms | FT 2.58x slower | 1.56x vs 8-iter baseline |

Score vs PyTorch: `0W / 1L / 0N`.
Internal score vs prior FT source: `1W / 0L / 0N`.

## Verification

- `cargo +nightly-2026-06-09 test -j4 --release -p ft-api prelu_backward_input_and_weight_and_value_parity --lib -- --nocapture` passed.
- `cargo +nightly-2026-06-09 test -j4 --release -p ft-conformance` passed: 199 lib tests plus bin, integration, smoke, and doctests.
- `cargo +nightly-2026-06-09 clippy -j4 --release -p ft-api --lib --example prelu_train_borrowed_h2h --no-deps -- -D warnings` passed.
- Full clippy without `--no-deps` is blocked by pre-existing `ft-kernel-cpu` warnings (`manual_memcpy`, `doc_lazy_continuation`, `manual_is_multiple_of`, `type_complexity`) outside this change.
- `rustfmt --check` passed for `crates/ft-api/examples/prelu_train_borrowed_h2h.rs`; checking all of `crates/ft-api/src/lib.rs` is blocked by unrelated pre-existing QR formatting drift.

## Disposition

Keep. This does not dominate PyTorch yet, but it is a material measured FT
train-step improvement and removes two dead full-tensor clones without changing
PReLU values or gradients. The remaining loss is not another context-save clone;
next work should target a fused PReLU train-step primitive or broader tensor
construction/session overhead.
