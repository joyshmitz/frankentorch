# Pass 6 Isomorphism Proof

## Change

`tensor_unfold` now allocates the coordinate scratch buffer once before building the gather table and overwrites every coordinate for each `flat_out`.

## Proof

- Ordering preserved: yes. The `flat_out in 0..out_numel` loop order is unchanged.
- Tie-breaking unchanged: yes. No comparisons or tie decisions are involved.
- Floating-point: identical. The change only affects integer coordinate scratch storage before gathering existing `f64` values.
- RNG seeds: unchanged. No RNG path is touched.
- Error diagnostics: unchanged. Shape, size, step, and overflow checks are before the changed loop and use the same messages.
- Forward mapping: unchanged. `coords[d]` is assigned before use for every `d`, and `in_flat` uses the same formula.
- Backward mapping: unchanged. The same `gather` table drives the scatter-add closure.
- Golden outputs: `sha256sum -c tests/artifacts/perf/20260602T004332Z-rustickite-pass6/golden_checksums.txt` passed.

## Validation

- `rch exec -- cargo test -p ft-api unfold`: 4 passed.
- `rch exec -- cargo test -p ft-api functional_conv2d_with_bias`: 1 passed.
- `rch exec -- cargo bench -p ft-api --bench ops_bench -- conv2d/hw/32 --warm-up-time 1 --measurement-time 5 --sample-size 20`: p50 220.58 ms after baseline p50 233.38 ms.
- `rch exec -- cargo check -p ft-api --all-targets`: passed with existing warnings.
- `rch exec -- cargo clippy -p ft-api --all-targets -- -D warnings`: failed on existing `ft-api` lint backlog outside this change.
- `rch exec -- cargo fmt -p ft-api --check`: failed on existing broad formatting drift in `ft-api`.
- `ubs crates/ft-api/src/lib.rs`: failed on existing monolith-wide UBS findings; no finding was introduced by the two-line scratch-buffer change.
- `git diff --check -- crates/ft-api/src/lib.rs tests/artifacts/perf/20260602T004332Z-rustickite-pass6`: passed.
