# frankentorch-fpzb proof report

## Target

- Bead: `frankentorch-fpzb`
- Target: `ft-kernel-cpu::pinv_qr_full_column_rank_f64` triangular backsolve used by no-grad full-rank `tensor_linalg_pinv`.
- Profile source: post-`frankentorch-0qwa` Criterion on RCH `ts1`; `lstsq/pinv_svd_512x256` still spent 165.81 ms median after SVD-to-QR routing.

## One lever

Replace the column-oriented triangular solve:

- old loop order: RHS column `c`, row `i`, solved row `jj`
- new loop order: row `i`, solved row `jj`, contiguous RHS element `c`

The QR factorization, reflector application, rank tolerance, fallback decisions, and output shape/order are unchanged.

## Benchmark

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- lstsq/pinv_svd --warm-up-time 1 --measurement-time 5 --sample-size 10
```

Same worker: `ts1`.

| benchmark | before median | after median | speedup |
| --- | ---: | ---: | ---: |
| `lstsq/pinv_svd_256x128` | 13.074 ms | 11.348 ms | 1.152x |
| `lstsq/pinv_svd_512x256` | 165.81 ms | 130.66 ms | 1.269x |

Score: Impact 3.0 x Confidence 0.90 / Effort 1.0 = 2.70.

## Isomorphism proof

- Ordering: each individual `pinv[i, c]` still subtracts solved rows in increasing `jj = i + 1..n` order before the final division. The outer traversal changed to make RHS rows contiguous, but the floating-point operation order for each result element is preserved.
- Tie-breaking: no pivoting, rank decision, tolerance, or comparison path changed.
- Floating point: same QR reflector values, same `r[i, jj]`, same `diag[i]`, same subtraction order per output element, same final division by `diag[i]`.
- RNG: no RNG path involved.
- Layout: output remains row-major `n * m` pseudo-inverse data.
- Fallback: rank-deficient, empty, and shape fallback behavior is unchanged.

## Verification

- `cargo test -p ft-kernel-cpu pinv_qr -- --nocapture`: pass.
- `cargo test -p ft-api pinv -- --nocapture`: pass.
- `cargo check -p ft-kernel-cpu -p ft-api --all-targets`: pass.
- `cargo clippy -p ft-kernel-cpu --all-targets --no-deps -- -D warnings`: pass.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`: pass.
- `git diff --check`: pass.
- `ubs crates/ft-kernel-cpu/src/lib.rs`: pass with 0 critical findings; remaining warnings are the existing whole-file inventory.
- Post-rebase on `origin/main` commit `16bfd082`: `cargo test -p ft-api pinv -- --nocapture`: pass.
- Post-rebase on `origin/main` commit `16bfd082`: `cargo check -p ft-kernel-cpu -p ft-api --all-targets`: pass.

Formatting note: `cargo fmt -p ft-kernel-cpu --check` reports pre-existing drift outside this change at lines 4930, 19054, 19074, 21376, 21385, 21410, and 21421. The touched backsolve block does not appear in the post-adjustment fmt diff.
