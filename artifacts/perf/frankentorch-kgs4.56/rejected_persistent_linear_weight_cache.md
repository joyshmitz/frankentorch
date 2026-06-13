# frankentorch-kgs4.56 rejection: persistent f64 Linear weight transpose cache

Bead: `frankentorch-kgs4.56`
Target: no-grad f64 `functional_linear` repeated-forward Criterion group
Worker: `vmi1227854`
Lever tested: session-local cache keyed by `(TensorNodeId, storage_id, version, in_features, out_features)` that stores `weight^T` and dispatches f64 Linear through normal `dgemm`.

## Baseline

Command:

```bash
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
  rch exec -- cargo bench -j 1 -p ft-api --bench ops_bench linear_forward/hidden -- \
  --warm-up-time 1 --measurement-time 3 --sample-size 10
```

Artifact: `baseline_ft_api_linear_forward.log`
SHA-256: `bcc49fe061e7a4c1f7cafdb2ceaeaa7c02b0a04ed39509f3f59d0127965b009d`

| case | median |
| --- | ---: |
| `linear_forward/hidden/256` | 230.35 us |
| `linear_forward/hidden/512` | 465.65 us |
| `linear_forward/hidden/1024` | 859.69 us |
| `linear_forward/hidden/2048` | 1.7880 ms |

## Candidate proof

RCH check:

```bash
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
  rch exec -- cargo check -j 1 -p ft-api --lib --benches
```

Result: pass on `vmi1227854`.

RCH focused tests:

```bash
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
  rch exec -- cargo test -j 1 -p ft-api functional_linear_f64_cached_weight -- --nocapture
```

Result: 2 passed on `vmi1227854`.

Isomorphism proof:

- Ordering: cold cached and hot cached f64 Linear outputs were compared bit-for-bit against explicit `weight.transpose(0, 1)` plus `addmm`.
- Invalidation: after an in-place `weight` version bump, cached Linear output was compared bit-for-bit against explicit transpose plus `matmul`.
- Tie-breaking: no comparisons or ordering decisions were introduced.
- Floating point: candidate changed only the materialization site for `weight^T`; all accepted-output proof used `f64::to_bits` equality.
- RNG: no RNG state was read or mutated by the cache. Test tensors were deterministic literal vectors.

## After

Artifact: `after_ft_api_linear_forward.log`
SHA-256: `4c6a9caf59cadd5c925230c35cb1bef515cd8bf223f95850f01cda73f6c525a3`

| case | baseline median | after median | speedup |
| --- | ---: | ---: | ---: |
| `linear_forward/hidden/256` | 230.35 us | 240.66 us | 0.957x |
| `linear_forward/hidden/512` | 465.65 us | 584.84 us | 0.796x |
| `linear_forward/hidden/1024` | 859.69 us | 799.91 us | 1.075x |
| `linear_forward/hidden/2048` | 1.7880 ms | 1.9574 ms | 0.913x |

Score: `< 2.0` because the candidate regressed three of four same-worker rows.

Decision: rejected and source patch removed.

Next primitive: microkernel-native persistent packed `dgemm_bt` weight panels for the skinny Linear shape family (`batch=32`, `in_features=512`, `out_features in {256,512,1024,2048}`), not a normal-GEMM transpose cache. Target ratio: at least 1.20x on `linear_forward/hidden/{1024,2048}` with no regression on smaller rows.
