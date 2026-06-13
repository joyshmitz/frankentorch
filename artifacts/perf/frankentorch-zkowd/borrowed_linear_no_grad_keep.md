# frankentorch-zkowd: f64 Linear no-grad borrowed inputs

Bead: `frankentorch-zkowd`

Lever kept: in `FrankenTorchSession::functional_linear`, the no-grad f64 fast
path borrows contiguous input, weight, and bias slices from `TensorTape` before
calling `ft_kernel_cpu::linear_tensor_f64`, instead of cloning those slices into
temporary `Vec`s. The compute primitive is still `linear_tensor_f64 ->
gemm::dgemm_bt`; only input ownership changes.

## Same-worker benchmark

Worker: `vmi1227854`

Command shape:

```text
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
rch exec -- cargo bench -j 1 -p ft-api --bench ops_bench linear_forward/hidden -- \
  --warm-up-time 1 --measurement-time 3 --sample-size 10
```

Criterion medians:

| Case | Baseline | After | Speedup |
| --- | ---: | ---: | ---: |
| `linear_forward/hidden/256` | 236.40 us | 198.70 us | 1.190x |
| `linear_forward/hidden/512` | 498.36 us | 438.43 us | 1.137x |
| `linear_forward/hidden/1024` | 1.1583 ms | 477.64 us | 2.425x |
| `linear_forward/hidden/2048` | 2.1266 ms | 900.78 us | 2.361x |

Score: Impact 3 x Confidence 3 / Effort 1 = 9.0. Keep.

## Transcript hashes

```text
3777e111baeee58a9381e08ade3f5ac4748c678da332ac9ed38dae83e24edc5f  baseline_ft_api_linear_forward.log
71b793c774b1ed8d6e266d45728b7897475764384a9a4f86313845cf5399ca4a  after_ft_api_linear_forward.log
4177cca512312bfabca7f3c70ef0caf1d05f4d63a1bde5e89dab675ceb0d8554  test_f64_linear_bit_exact.log
```

## Behavior proof

- Arithmetic order: unchanged. Both before and after call
  `ft_kernel_cpu::linear_tensor_f64(x, weight, bias, batch, in_features,
  out_features)`, which computes `gemm::dgemm_bt(batch, in_features,
  out_features, x, weight, y)` and then adds bias by row.
- Tensor ordering: unchanged. The borrowed slices come from
  `contiguous_values()` on the same `input`, `weight`, and `bias` nodes that the
  old `tensor_values()` clones read.
- Tie-breaking: not applicable; no comparisons or reductions with ties were
  introduced.
- Floating point: unchanged. No arithmetic, associativity, threading threshold,
  alpha/beta scaling, or bias-add order changed.
- RNG: not applicable; this path is deterministic and uses no random state.
- Autograd: unchanged. This is inside the no-grad fast path only; grad-required
  f64 Linear continues through the existing borrowed-input autograd branch.
- Golden-output check: retained RCH test
  `functional_linear_f64_no_grad_matches_transpose_path_bit_exact` compares the
  fused borrowed path against explicit `weight.transpose(0, 1) + tensor_addmm`
  by `f64::to_bits()` for every output element. Transcript SHA-256:
  `4177cca512312bfabca7f3c70ef0caf1d05f4d63a1bde5e89dab675ceb0d8554`.

## Validation

Passed:

```text
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
rch exec -- cargo check -j 1 -p ft-api --lib --benches

RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
rch exec -- cargo test -j 1 -p ft-api \
  functional_linear_f64_no_grad_matches_transpose_path_bit_exact -- --nocapture
```

Blocked by existing project debt, not by this lever:

```text
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
rch exec -- cargo clippy -j 1 -p ft-api --lib --benches -- -D warnings
```

The clippy run failed with the existing broad `ft-api` lint inventory (256
warnings promoted to errors across distant sections of `lib.rs`, including
`manual_is_multiple_of`, `needless_range_loop`, `doc_lazy_continuation`, and
`excessive_precision`). The perf commit does not expand into that unrelated
cleanup.

`cargo fmt -p ft-api --check` is likewise blocked by pre-existing formatting
diffs in distant `ft-api` examples/tests; no package-wide formatting was
applied.
