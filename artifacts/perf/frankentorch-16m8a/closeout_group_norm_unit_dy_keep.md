# frankentorch-16m8a closeout: GroupNorm sum-gradient backward keep

Target:
- `group_norm/grad_32x256x28x28`
- Profile source: `artifacts/perf/frankentorch-next-reprofile-20260617b/current_top_train_reprofile_after_6olvt.log`
- Reprofile timing after `frankentorch-6olvt`: `[480.19 ms 489.97 ms 500.44 ms]`
- Dedicated baseline: `[525.34 ms 534.80 ms 543.59 ms]`

One lever:
- Add an exact f64 GroupNorm backward path for all-ones upstream gradients.
- The branch stages per-`(batch, group)` `(mean, rstd)` once inside backward, uses those stats for `dx`, then preserves the old serial group/element order for affine `dweight` and `dbias`.
- Non-one `dy` and f32 paths fall through unchanged.

Isomorphism proof:
- Guard: `dy.to_bits() == 1.0f64.to_bits()` for every upstream element.
- Ordering: `dx` still writes each group row in ascending element order; affine gradients retain old serial group-major then element-major accumulation.
- Floating point: per-group stat reductions use the same element order as the old recomputing path; kernel proof compares `dx`, `dweight`, and `dbias` bit-for-bit against the old formula.
- RNG/tie behavior: no RNG, tie-breaking, or shape/error behavior is introduced.
- Golden digest: `0x533d76b2bc4e86e3`.

Benchmark result:
- Baseline: `[525.34 ms 534.80 ms 543.59 ms]`.
- First candidate: `[506.24 ms 514.85 ms 524.07 ms]`, Criterion change `[-5.8976% -3.7291% -1.2995%]`, `p = 0.01`.
- Final candidate after clippy loop spelling fix: `[503.54 ms 513.67 ms 522.40 ms]`.
- Final raw median speedup: `1.0411x` (`534.80 ms -> 513.67 ms`).
- Final baseline/candidate intervals do not overlap.

Score:
- Impact: `1.041`
- Confidence: `0.95`
- Effort: `0.45`
- Score: `2.20`
- Decision: keep.

Validation:
- `cargo test -j 1 -p ft-kernel-cpu group_norm_f64_unit_dy_matches_general_reference_bits -- --nocapture` passed.
- `cargo test -j 1 -p ft-api functional_group_norm_grad_matches_finite_diff -- --nocapture` passed.
- `cargo check -j 1 -p ft-kernel-cpu --all-targets` passed; it still reports pre-existing example warnings in `crates/ft-kernel-cpu/examples/gemm_golden.rs`.
- `cargo clippy -j 1 -p ft-kernel-cpu --lib -- -D warnings` passed.
- `git diff --check` passed.
- `rustfmt --edition 2024 --check crates/ft-kernel-cpu/src/lib.rs` still reports pre-existing broad file drift outside this patch; no whole-file formatting rewrite was applied.
- `ubs crates/ft-kernel-cpu/src/lib.rs` completed with no critical findings; broad pre-existing warning inventory remains.

Artifact hashes:
- `pass1_local_baseline_group_norm_grad.log`: `6d8b658eb35cba477f07b75f3d996ce71d62a9d4c8b45fb01a85c108b33217cd`
- `pass2b_kernel_proof_after_clippy_fix.log`: `f53d3c159dee2401290e4efd2db6e023a6a69a38b39854ab6922b0313e37fa26`
- `pass2b_api_group_norm_grad_test_after_clippy_fix.log`: `72d127aa44cd5e0a8ab95442e50205c4286b962c545fd909c1314eeb959497be`
- `pass3b_local_rebench_group_norm_unit_dy_final.log`: `f2f154a2bed34094d28526a423114740c0865de777e2894a2d103e5742bea368`
- `pass4b_check_ft_kernel_cpu_all_targets_after_fix.log`: `3372bef49d5581e188ae55378beccd76396f68e781266b637df228e009180830`
- `pass4b_clippy_ft_kernel_cpu_lib_after_fix.log`: `0058dda5fe1474a15bb4fb4de4fd866d51eb1b9fcf68ae2588bd3d30e926feae`
- `pass4b_git_diff_check.log`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `pass4b_rustfmt_ft_kernel_cpu_check.log`: `fd6e3aab93d9d1d7d9c23a2da244ef1f60e235a211f8db219861f662873413f4`
- `pass4_ubs_ft_kernel_cpu.log`: `61f7877d757c88f73dc24727e75706fd18df2f6c47c89c670659f39bd3665bb6`
