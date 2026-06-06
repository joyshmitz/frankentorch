# frankentorch-3oht proof report

## Target

- Bead: `frankentorch-3oht`
- Target: `ft-kernel-cpu::pinv_qr_full_column_rank_f64` `q_t_rows` Householder replay.
- Profile source: post-`frankentorch-fpzb` RCH `ts1` Criterion showed direct QR least-squares at 82.664 ms median while full pseudo-inverse materialization was 125.84 ms median for the 512x256 family.
- Alien primitive: CA-QR / blocked Householder replay guidance from the alien graveyard; apply support-structure knowledge before deeper blocked kernels.

## One lever

During `q_t_rows` reflector replay, skip rows `< j` for finite reflectors. By induction those rows are exact zero in the affected `j..m` segment before reflector `j` is applied. If the reflector or scale is non-finite, the implementation falls back to the old all-row replay to preserve NaN propagation.

## Benchmark

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- lstsq/pinv_svd --warm-up-time 1 --measurement-time 5 --sample-size 10
```

Same worker: `ts1`.

| benchmark | before median | after median | speedup |
| --- | ---: | ---: | ---: |
| `lstsq/pinv_svd_256x128` | 10.526 ms | 8.4321 ms | 1.248x |
| `lstsq/pinv_svd_512x256` | 125.84 ms | 106.41 ms | 1.183x |

Score: Impact 3.0 x Confidence 0.90 / Effort 1.0 = 2.70.

## Isomorphism proof

- Ordering: rows that still execute use the same row order, same dot-product order over `i`, and same update order over `i` as before.
- Skipped rows: for finite reflectors, rows `< j` are exact zero in columns `j..m`; the old loop computed `dot = 0.0`, `f = 0.0`, and wrote no meaningful value change.
- Non-finite behavior: if `inv` or any reflector element is non-finite, `first_active_row` is `0`, so the old all-row path is used and NaN propagation is preserved.
- Tie-breaking: no rank tolerance, sign, pivot, or fallback decision changed.
- Floating point: no arithmetic order changes for nonzero affected rows; zero-prefix elision only removes finite zero work.
- RNG: no RNG path involved.
- Layout: output remains row-major `n * m` pseudo-inverse data.

## Verification

- `cargo test -p ft-kernel-cpu pinv_qr -- --nocapture`: pass.
- `cargo test -p ft-api pinv -- --nocapture`: pass.
- `cargo check -p ft-kernel-cpu -p ft-api --all-targets`: pass.
- `cargo clippy -p ft-kernel-cpu --all-targets --no-deps -- -D warnings`: pass.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`: pass.
- `git diff --check`: pass.
- `ubs crates/ft-kernel-cpu/src/lib.rs`: pass with 0 critical findings; remaining warnings are the existing whole-file inventory.

Formatting note: `cargo fmt -p ft-kernel-cpu --check` reports pre-existing drift outside this change at lines 4930, 19059, 19079, 21381, 21390, 21415, and 21426. The touched reflector replay block does not appear in the fmt diff.
