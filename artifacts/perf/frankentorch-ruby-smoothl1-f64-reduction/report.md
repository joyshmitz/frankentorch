# smooth_l1 f64 no-grad reduction map-reduce

Bead: `frankentorch-lonz`
Measured base: `2735f8d6ad9fd284833fe01ecc35ad0e670d735a`
Rebased parent: `a5332e3c5399f1970b34b0a138820a851a89ee64`
Agent: `RubyLotus`

## Target

`br ready --json --no-auto-import` had no ready work; current in-progress perf lanes were owned by other agents. I selected the disjoint existing Criterion target `smooth_l1/nograd_8m`, which profiles the same-shape f64 no-grad `mean` path. The prior f64 fast path still materialized the full per-element smooth_l1 vector, then called `tensor_mean` or `tensor_sum`.

Primitive harvested: fused pairwise map-reduce / materialization removal. This is a structural change: compute each smooth_l1 value at the reduction leaves and preserve the existing midpoint pairwise reduction tree instead of allocating and then reducing an 8M-element loss tensor.

## Benchmark

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-api --bench ops_bench -- smooth_l1/nograd_8m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Same worker: `ts1`

The final commit was rebased after three unrelated upstream commits landed (`cross_entropy` borrowed-input grad, shared half dtype cat/stack, and half dtype `index_select` autograd). Those commits do not touch the smooth_l1 reducer or the `smooth_l1/nograd_8m` benchmark body; focused smooth_l1 tests were rerun on the rebased source.

| Run | Criterion interval |
| --- | --- |
| Baseline current tip | `[131.50 ms 136.80 ms 142.52 ms]` |
| Candidate | `[95.140 ms 97.302 ms 100.17 ms]` |

Median speedup: `136.80 / 97.302 = 1.41x`.

Score: `Impact 3.0 x Confidence 0.95 / Effort 1.1 = 2.59`, keep.

## Isomorphism Proof

Ordering and tie-breaking:
The optimized route is only for same-shape f64 no-grad `mean` and `sum`. The `none` reduction keeps the old per-element tensor path, so output element ordering is unchanged there. Scalar reductions have no tie-breaking surface.

Floating point:
The per-element expression is factored into `smooth_l1_value_f64`, used by both the old materialized `smooth_l1_forward_f64` path and the new reducer. The reducer splits at the same `mid = len / 2` boundaries, uses the same 128-element serial leaves and the same parallel threshold/tree shape as `sum_tensor_contiguous_f64`, and combines every node as `left + right`. `mean` divides the resulting sum by the same `n as f64`. Focused tests assert `to_bits()` equality for sum and mean against the materialized-vector path.

RNG:
No RNG is introduced. The benchmark creates tensors before `b.iter`; the optimized path only reads tensor values and computes a deterministic scalar.

Autograd:
The fast path is gated on both input and target not requiring gradients. Any gradient-bearing input still uses the existing custom autograd path. The existing finite-difference smooth_l1 test still passes.

Shape and dtype:
The API returns the same scalar tensor shape `[1]`, dtype `DType::F64`, and `requires_grad = false` for no-grad `mean` and `sum`. The API parity test asserts these properties and bit-exact values.

## Gates

- `cargo test -p ft-kernel-cpu smooth_l1_reduced_f64 -- --nocapture`: passed, 2 tests.
- `cargo test -p ft-api smooth_l1_loss -- --nocapture`: passed, 8 tests.
- `cargo check -p ft-api -p ft-kernel-cpu --all-targets`: passed; three pre-existing `unused_mut` warnings in recurrent tests remain.
- `cargo fmt --check`: failed on broad pre-existing formatting drift across benches/crates; no formatter was applied in this lane.
- `cargo clippy -p ft-api -p ft-kernel-cpu --all-targets -- -D warnings`: failed on existing `ft-api` lint debt, ending with 207 errors unrelated to `smooth_l1`.
- `ubs crates/ft-api/src/lib.rs crates/ft-kernel-cpu/src/lib.rs`: completed; reported broad existing inventories across the two large files. No reducer-specific finding was surfaced by the focused proof tests.
- `git diff --check`: passed.

## Evidence

See:

- `baseline_2735f8d6.txt`
- `after_2735f8d6_candidate.txt`
- `test_ft_kernel_cpu_smooth_l1_reduced.txt`
- `test_ft_api_smooth_l1_loss.txt`
- `check_ft_api_kernel.txt`
- `fmt_check.txt`
- `clippy_ft_api_kernel.txt`
- `ubs_ft_api_kernel.txt`
- `evidence.sha256`
