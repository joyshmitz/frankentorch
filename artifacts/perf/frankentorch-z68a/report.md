# frankentorch-z68a: rejected parallel Householder row replay

## Target

- Target: `ft-kernel-cpu::pinv_qr_full_column_rank_f64` `q_t_rows` Householder replay.
- Profile source: after `frankentorch-3oht`, RCH `ts1` Criterion `lstsq/pinv_svd_512x256` remained about 104-105 ms median while direct QR least-squares was 82.664 ms median.
- Lever attempted: parallelize independent `q_t_rows` rows for large reflector replay panels, matching the row fan-out pattern already used by `qr_contiguous_f64`.

## Benchmark

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- lstsq/pinv_svd --warm-up-time 1 --measurement-time 5 --sample-size 10
```

| Benchmark | Baseline median | Parallel median | Result |
| --- | ---: | ---: | ---: |
| `lstsq/pinv_svd_256x128` | 7.8764 ms | 9.7198 ms | 0.810x |
| `lstsq/pinv_svd_512x256` | 105.01 ms | 168.82 ms | 0.622x |

Score: rejected. The top size regressed badly, so the lever did not clear `Score >= 2.0`.

## Isomorphism

The candidate preserved per-row dot/update order, reflector order, rank tolerance, finite/non-finite fallback, output ordering, and RNG. No candidate code was kept, so current source behavior is exactly the pre-z68a implementation.

## Next Primitive

Do not continue Rayon fan-out or reflector-storage micro-levers in this lane. The next pass should attack a different structural primitive: a blocked/WY pseudo-inverse materialization or a normal-equations/Cholesky fast path with strict condition guards and SVD fallback, benchmarked separately against the current QR pseudo-inverse path.
