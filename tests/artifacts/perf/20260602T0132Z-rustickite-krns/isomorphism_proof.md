# frankentorch-krns Isomorphism Proof

## Lever

`tensor_unfold` now advances the output coordinate vector incrementally in
row-major order while building the gather table. This removes per-output
division and modulo coordinate decoding.

## Preserved Semantics

- Output ordering: unchanged. The coordinates advance in the same row-major
  order as the previous `flat_out / out_strides[d]` decode.
- Tie-breaking: not applicable. No comparisons or ordering decisions changed.
- Floating-point behavior: unchanged. The forward still copies `vals[k]`; the
  backward still accumulates `grad_in[in_flat] += g[flat_out]` in the same
  iteration order.
- RNG behavior: unchanged. No random state is read or advanced.
- Diagnostics and shape behavior: unchanged. Validation, overflow checks, output
  shape construction, and error strings are untouched.
- Aliasing and mutation: only the local coordinate scratch buffer mutates while
  building the immutable gather table. No tensor data aliasing changes.

## Golden Output

`sha256sum -c tests/artifacts/perf/20260602T0132Z-rustickite-krns/golden_checksums.txt`
passed for `dbd74c3723fcd8628e7c91b58b3676e476ea7858ed9b2bb7d6735690fa4e04db`.

## Validation

- `sha256sum -c tests/artifacts/perf/20260602T0132Z-rustickite-krns/golden_checksums.txt`: passed.
- `git diff --check -- crates/ft-api/src/lib.rs tests/artifacts/perf/20260602T0132Z-rustickite-krns .skill-loop-progress.md`: passed.
- `rch exec -- cargo test -p ft-api unfold`: passed, 4 tests.
- `rch exec -- cargo test -p ft-api functional_conv2d_with_bias`: passed, 1 test.
- `rch exec -- cargo check -p ft-api --all-targets`: passed with existing warnings.
- `rch exec -- cargo bench -p ft-api --bench ops_bench -- conv2d/hw/32 --warm-up-time 1 --measurement-time 5 --sample-size 20`: p50 180.02 ms after same-worker baseline p50 206.53 ms.
- `rch exec -- cargo clippy -p ft-api --all-targets -- -D warnings`: failed on the existing ft-api lint backlog, 89 errors, with first unrelated findings at lines 2090, 4933, and 58867.
- `rch exec -- cargo fmt -p ft-api --check`: failed on existing broad ft-api formatting drift in `ops_bench.rs` and unrelated `src/lib.rs` regions.
- `ubs crates/ft-api/src/lib.rs`: failed on existing monolith-wide scanner findings: 311 critical comparison-pattern findings, 16101 warnings, and 1720 info items. UBS reported no unsafe blocks.
