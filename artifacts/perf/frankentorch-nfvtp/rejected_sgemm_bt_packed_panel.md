# frankentorch-nfvtp: rejected packed-panel sgemm_bt

Bead: `frankentorch-nfvtp`

Attempted lever: for the f32 `sgemm_bt` column-parallel path, pack each selected
BT column block into a contiguous logical `[k, bw]` panel before calling the
existing SGEMM microkernel.

Decision: reject. No source change is retained.

## Baseline

Worker: `vmi1152480`

Command:

```text
RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
rch exec -- cargo bench -j 1 -p ft-api --bench ops_bench linear_forward/f32_hidden -- \
  --warm-up-time 1 --measurement-time 3 --sample-size 10
```

Criterion medians:

| Case | Baseline |
| --- | ---: |
| `linear_forward/f32_hidden/1024` | 983.68 us |
| `linear_forward/f32_hidden/2048` | 1.5119 ms |

## Candidate validation

With the temporary packed-panel edit applied:

```text
RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
rch exec -- cargo check -j 1 -p ft-kernel-cpu --lib --benches

RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
rch exec -- cargo test -j 1 -p ft-kernel-cpu \
  sgemm_bt_col_packed_is_bit_exact_vs_serial -- --nocapture
```

Both passed. The focused test compared packed-panel BT column output against
serial `sgemm_bt_block` by `f32::to_bits()` for every output element.

## Same-worker after

Worker: `vmi1152480`

| Case | Baseline | After | Ratio |
| --- | ---: | ---: | ---: |
| `linear_forward/f32_hidden/1024` | 983.68 us | 1.4980 ms | 0.657x |
| `linear_forward/f32_hidden/2048` | 1.5119 ms | 1.8227 ms | 0.829x |

Score: 0. Reject. The panel pack is bit-exact but the pack cost exceeds any B
access locality win for this small-M/wide-N f32 Linear shape.

## Transcript hashes

```text
3b3456d5e16fad90185efc2b687d72864c7838a61d8a5891aa2190d14ca0bb1c  baseline_ft_api_linear_f32_forward.log
967d9e5d767110b6e0b63c83edb89eeb374d7218d7459b506947074c4b4e9742  check_ft_kernel_cpu_lib_benches.log
e4064dc2990e21e51b84c201da43a16bfcccddbec48a96417b6bb960468db957  test_sgemm_bt_col_packed_bit_exact.log
fd1a9674671556830ebdb19560a9a258a77cd5509d642c00ae8f4a23a9d32bf9  after_ft_api_linear_f32_forward.log
```

## Isomorphism

The temporary packed-panel path preserved the same public shape checks, output
layout, column-window partitioning, K accumulation order under the existing
microkernel, tie behavior, and RNG behavior. The bit-exact focused test passed,
but performance regressed, so the source edit was removed.

## Next route

Do not continue per-call panel packing for this shape. A deeper f32 Linear route
should avoid extra per-call packing, for example by attacking register blocking
inside the current stride-BT microkernel path or by introducing persistent
weight-layout transforms with explicit invalidation and same-worker proof.
