# BMM Batch GEMM Optimization Pass 13

Bead: `frankentorch-9hlj`

## Profile Target

- Target: `ft-api` Criterion benchmark `bmm/batch/32`.
- Baseline command: `CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-pass12-target rch exec -- cargo bench -p ft-api --bench ops_bench -- bmm/batch/32 --warm-up-time 1 --measurement-time 5 --sample-size 20`
- Baseline worker/result: `vmi1264463`, time `[9.9704 ms 16.042 ms 25.411 ms]`.
- Dependent profile context: earlier `conv2d/hw/32` baseline on `vmi1227854` was `[131.60 ms 135.32 ms 139.40 ms]`, but a concurrent in-progress `ft-api` conv2d bead (`frankentorch-hx4y`) owns that surface, so this pass gates on direct BMM only.

## Lever

Parallelize independent f64 BMM batch GEMMs only for larger batch counts:

- `batch < 8`: preserve the serial loop to avoid Rayon overhead on small batches.
- `batch >= 8`: write each output batch through `par_chunks_exact_mut(out_batch_stride)`.
- Slice each lhs/rhs batch to its exact batch window instead of passing the whole suffix.
- Leave f32 BMM unchanged.

## Isomorphism Proof

- Ordering: `enumerate()` maps each parallel output chunk to the same batch index and same output slice as the serial loop.
- Floating point: each batch still invokes the same `gemm::dgemm(m, k, n, ...)` with the same lhs/rhs values; no cross-batch reduction is introduced.
- Tie-breaking: not applicable; BMM has no ordering-dependent ties beyond per-batch GEMM arithmetic, which is unchanged.
- RNG: unchanged; no random source is read or reordered.
- Diagnostics: rank, dtype, contiguity, storage bounds, and shape checks still run before the parallel branch.
- Empty output: returns the same empty output after validation when `out_batch_stride == 0`.
- Golden output: `artifacts/optimization/golden_outputs/bmm_batch_pass13.txt`, sha256 `061cf87277276c6d5dab0d439c027f7008862c968cf2f62720d6e2f42f2d6441`.

## Validation

- `rch exec -- cargo test -p ft-kernel-cpu bmm_tensor_contiguous -- --nocapture`: passed, 3 tests.
- `rch exec -- cargo check -p ft-kernel-cpu --all-targets`: passed.
- `rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`: passed.
- `rch exec -- cargo fmt -p ft-kernel-cpu --check`: failed on pre-existing file-wide rustfmt drift outside this pass; pass-local `git diff --check -- crates/ft-kernel-cpu/src/lib.rs` passed.
- `ubs crates/ft-kernel-cpu/src/lib.rs`: exit 0; no critical findings, pre-existing warning inventories remain.
- `sha256sum -c artifacts/optimization/golden_checksums.txt`: passed; all listed golden outputs verified, including `bmm_batch_pass13.txt`.

## Result

- After command: `CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-9hlj-threshold-target rch exec -- cargo bench -p ft-api --bench ops_bench -- bmm/batch/32 --warm-up-time 1 --measurement-time 5 --sample-size 20`
- After worker/result: `vmi1149989`, time `[1.0243 ms 1.1265 ms 1.2016 ms]`.
- Improvement: mean `16.042 ms -> 1.1265 ms`, about `14.2x` faster on the direct profile target.
- Score: impact 3 x confidence 3 / effort 2 = 4.5.
- Verdict: keep and close.
