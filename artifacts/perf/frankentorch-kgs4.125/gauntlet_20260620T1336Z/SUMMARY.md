# frankentorch-kgs4.125 BatchNorm1d NCL Gauntlet

Date: 2026-06-20
Agent: IvoryDeer / cod-a

## Verdict

Keep. Native `[N,C,L]` BatchNorm1d is a large measured internal win over the
explicit fold route. The follow-up BatchNorm row-task coarsening is a smaller
same-host keep. The workload still loses to PyTorch, so this is not a domination
claim.

## Measurements

- RCH pre-coarsening Criterion on `vmi1227854`:
  - native NCL median: `4.3741 ms`
  - fold-reference median: `30.484 ms`
  - native/fold: `0.1435x`, or `6.97x` faster
- Local same-host pre-coarsening Criterion:
  - native NCL median: `11.865 ms`
  - fold-reference median: `60.554 ms`
  - native/fold: `0.1959x`, or `5.10x` faster
- Local PyTorch CPU oracle:
  - torch `2.12.1+cpu`
  - median: `2.251326 ms`
  - pre-coarsening FT/PyTorch: `5.27x` slower
- Local same-host after `BATCH_NORM_MIN_PAR_ROWS = 8`:
  - native NCL median: `10.914 ms`
  - fold-reference median: `57.450 ms`
  - native/fold: `0.1900x`, or `5.26x` faster
  - row coarsening native speedup: `1.09x`
  - after-coarsening FT/PyTorch: `4.85x` slower
- Supplemental RCH after coarsening on `hz1`:
  - native NCL median: `6.2713 ms`
  - fold-reference median: `60.234 ms`
  - native/fold: `0.1041x`, or `9.60x` faster
  - Not used as before/after proof because the worker differs from the RCH
    pre-coarsening run.

## Gates

- `cargo test -p ft-kernel-cpu batch_norm --lib -- --nocapture`: passed via RCH.
- `cargo test -p ft-api functional_batch_norm1d_3d_native_fused_matches_fold_reference_bits --lib -- --nocapture`: passed before and after row coarsening via RCH.
- `cargo check -p ft-api --benches`: passed via RCH.
- `cargo check -p ft-kernel-cpu --lib`: passed via RCH.
- `cargo clippy -p ft-kernel-cpu --lib -- -D warnings`: passed via RCH.
- `cargo clippy -p ft-api --bench ops_bench -- -D warnings`: first run exposed two pre-existing bench `single_element_loop` findings; fixed and rerun passed via RCH. After the UBS label-comparison cleanup, the bench clippy gate passed again.
- `cargo test -p ft-conformance`: RCH had no admissible workers and fell back local; full suite passed.
- `ubs <scoped files>`: initial run flagged the benchmark label equality as a constant-time comparison false positive; after switching that benchmark check to `.eq()`, the rerun reported 0 critical findings and preserved the existing broad warning inventory.
- `rustfmt --edition 2024 --check crates/ft-kernel-cpu/src/lib.rs`: passed.
- `rustfmt --edition 2024 --check crates/ft-api/benches/ops_bench.rs`: still blocked by unrelated pre-existing formatting drift elsewhere in the file.

## Files

- `criterion_batch_norm1d_ncl.log`
- `local_criterion_batch_norm1d_ncl.log`
- `pytorch_batch_norm1d_ncl_f64.log`
- `criterion_batch_norm1d_ncl_after_min_rows.log`
- `local_criterion_batch_norm1d_ncl_after_min_rows.log`
- `ft_kernel_cpu_batch_norm_after_min_rows.log`
- `ft_api_batch_norm1d_ncl_bits_after_min_rows.log`
- `check_ft_api_benches_after_min_rows.log`
- `check_ft_kernel_cpu_lib_after_min_rows.log`
- `clippy_ft_kernel_cpu_lib_after_min_rows.log`
- `clippy_ft_api_ops_bench_after_single_loop_fix.log`
- `clippy_ft_api_ops_bench_after_ubs_label_fix.log`
- `test_ft_conformance_after_min_rows.log`
- `ubs_scoped_after_label_fix.log`
