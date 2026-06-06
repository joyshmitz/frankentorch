# frankentorch-4niv: rejected packed Householder reflector slab

## Target

- Target: `ft-kernel-cpu::pinv_qr_full_column_rank_f64` stored Householder reflector layout.
- Profile source: after `frankentorch-3oht`, RCH `ts1` Criterion `lstsq/pinv_svd_512x256` still spent 104-106 ms median while direct QR least-squares was 82.664 ms median.
- Lever attempted: replace `Vec<Vec<f64>>` reflector storage with one contiguous packed slab plus per-reflector offsets.

## Benchmark

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- lstsq/pinv_svd --warm-up-time 1 --measurement-time 5 --sample-size 10
```

| Benchmark | Baseline median | Packed median | Result |
| --- | ---: | ---: | ---: |
| `lstsq/pinv_svd_256x128` | 8.2501 ms | 7.8573 ms | 1.050x |
| `lstsq/pinv_svd_512x256` | 104.10 ms | 104.86 ms | 0.993x |

Score: rejected. The larger, top-scored size regressed, so the lever did not clear `Score >= 2.0`.

## Isomorphism

The candidate preserved reflector construction, reflector application order, rank tolerance, finite/non-finite fallback, pseudo-inverse row order, and RNG behavior. No candidate code was kept, so current source behavior is exactly the pre-4niv `frankentorch-3oht` implementation.

## Next Primitive

The next pass should not continue reflector-storage micro-tuning. Attack a structural direct-pinv primitive instead: compute the pseudo-inverse with a blocked/WY Householder replay or direct row-panel application that reduces full `Q^T` materialization overhead while preserving the observable `tensor_linalg_pinv` output.
