# ft-kernel-cpu F32 TopK Bounded Selection - frankentorch-cltj

## Target

- Bead: `frankentorch-cltj`
- Crate: `ft-kernel-cpu`
- Benchmark: `topk_f32_8192x1024_k50_dim1`
- Command: `rch exec -- cargo bench -p ft-kernel-cpu --bench topk_bench -- topk_f32_8192x1024_k50_dim1 --warm-up-time 1 --measurement-time 5 --sample-size 20`
- Worker: `ts2`

## Baseline And Profile

- Baseline: `[9.6478 ms 10.286 ms 11.255 ms]`
- Profile-backed hotspot: the f32 topk path still full-sorted each 1024-element lane before taking `k=50`, while the f64 path already used bounded selection.
- Selected primitive: deterministic bounded top-k lane selection with `select_nth_unstable_by`, followed by ordering only the selected prefix.

## Lever

`topk_tensor_contiguous_f32` now uses the same bounded-selection structure as f64 for `k < dim_size`. The comparator includes the value ordering and original index, so unstable partitioning cannot change the selected set for equal values or NaNs. The `k == dim_size` path still full-sorts the lane.

## Result

- Candidate: `[4.7476 ms 4.9236 ms 5.1200 ms]`
- p50 speedup: `10.286 ms / 4.9236 ms = 2.089x`
- Score: `Impact 2.09 * Confidence 1.0 / Effort 1.0 = 2.09`
- Verdict: keep.

## Isomorphism Proof

- Ordering preserved: yes. Sorted output still follows value order with original-index tie order; unsorted output still follows original index order among selected elements.
- Tie-breaking preserved: yes. The selector comparator orders ties by original lane index, matching the previous stable full-sort observable result.
- Floating-point preserved: yes. The lever performs no arithmetic and copies the original f32 bits, including NaN payloads and signed zeros.
- NaN behavior preserved: yes. `nan_greatest_cmp_f32` remains the value comparator and NaNs sort largest.
- RNG unchanged: yes. No random state is read or modified.
- Shape/error behavior unchanged: yes. Input validation, output shape, index type, and `k == dim_size` behavior are unchanged.
- Golden output: `artifacts/optimization/golden_outputs/ft_kernel_cpu_topk_f32_bounded_selection_frankentorch-cltj.txt`
- Golden sha256: `373d0a1b799b73c412f668fb0e02a7ed536ee6c95bcbf2cda0b9aedfa037ab1c`

## Validation

- Focused proof test: `rch exec -- cargo test -p ft-kernel-cpu topk_f32_bounded_selection_matches_full_sort_bit_exact -- --nocapture` passed on `ts2`.
- Golden manifest: `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed.
- Crate check: `rch exec -- cargo check -p ft-kernel-cpu --all-targets` passed on `ts2`.
- Crate clippy: `rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings` passed on `ts2`.
- Crate rustfmt: `cargo fmt -p ft-kernel-cpu --check` passed.
- Diff hygiene: `git diff --check` passed for the touched files.
- UBS: `ubs crates/ft-kernel-cpu/src/lib.rs crates/ft-kernel-cpu/benches/topk_bench.rs artifacts/optimization/golden_outputs/ft_kernel_cpu_topk_f32_bounded_selection_frankentorch-cltj.txt artifacts/optimization/golden_checksums.txt` completed with 0 critical findings; remaining warnings are the existing file-wide inventory.
