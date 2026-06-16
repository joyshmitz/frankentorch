# frankentorch-a7xya closeout

Status: KEEP

## Change

Specialized f64 BatchNorm `spatial == 1` apply and `dx` phases to schedule by batch row (`par_chunks_mut(channels)`) instead of one Rayon chunk per scalar (`par_chunks_mut(spatial)`).

This is one lever. The existing `dweight`/`dbias` reduction path and its summation order are unchanged.

## Baseline And Rebench

Same-worker RCH Criterion target: `cargo bench -j 1 -p ft-api --bench ops_bench -- batch_norm/grad_1d_8192x1024 --warm-up-time 1 --measurement-time 5 --sample-size 20 --noplot`

- Worker: `vmi1149989`
- Baseline: `[513.28 ms 533.39 ms 554.63 ms]`
- Candidate: `[473.56 ms 491.76 ms 513.40 ms]`
- Median delta: `533.39 ms -> 491.76 ms`, `1.085x`
- Score: `2.06 = Impact 1.085 * Confidence 0.95 / Effort 0.50`

Evidence:

- `artifacts/perf/frankentorch-a7xya/pass1_baseline_batch_norm_grad_1d.log`
- `artifacts/perf/frankentorch-a7xya/pass4_rebench_batch_norm_grad_1d.log`

## Isomorphism Proof

- Ordering preserved: yes. Each output element and `dx` element is written once to the same index; only outer chunk granularity changes for `spatial == 1`.
- Tie-breaking unchanged: N/A.
- Floating-point behavior: bit-identical proof for apply, `dx`, `dweight`, and `dbias` on the spatial-1 fixture. Per-element arithmetic order is unchanged; `dweight`/`dbias` reduction order is unchanged.
- RNG seeds: N/A.
- Golden output: FNV digest `0x3cad78ffcd28e1f5` in `batch_norm_f64_spatial1_row_parallel_matches_serial_reference_bits`.

Proof logs:

- `artifacts/perf/frankentorch-a7xya/pass3_kernel_batch_norm_spatial1_golden.log`
- `artifacts/perf/frankentorch-a7xya/pass5_api_batch_norm1d_grad_test.log`

## Closeout Gates

- `rch exec -- cargo test -j 1 -p ft-kernel-cpu batch_norm_f64_spatial1_row_parallel_matches_serial_reference_bits --lib -- --nocapture`: pass on `vmi1156319`.
- `rch exec -- cargo check -j 1 -p ft-kernel-cpu --all-targets`: pass on `ovh-a`; existing `examples/gemm_golden.rs` unused-parens warnings remain unrelated.
- `rch exec -- cargo clippy -j 1 -p ft-kernel-cpu --lib --no-deps -- -D warnings`: pass on `ovh-a`.
- `git diff --check -- crates/ft-kernel-cpu/src/lib.rs`: pass.
- `ubs crates/ft-kernel-cpu/src/lib.rs`: exit 0; broad pre-existing warning inventory only.

`cargo fmt -p ft-kernel-cpu -- --check` remains blocked by existing formatting drift in unrelated examples and older source ranges; touched diff is whitespace-clean.
