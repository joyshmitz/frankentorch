# frankentorch-isc4: einsum RHS-transpose dgemm_bt fast path

## Target

- Bead: `frankentorch-isc4`
- Profile-backed source: `frankentorch-bkt6` showed einsum contraction work was blocked by materialized `tensor_permute` overhead, not contraction ordering.
- One lever: for no-grad f64 binary contractions shaped as `free_lhs,contract` by `free_rhs,contract` with output `free_lhs,free_rhs`, call a safe public `ft-kernel-cpu` wrapper over `dgemm_bt` instead of materializing the RHS transpose and then calling matmul.

## Benchmark

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- einsum/rhs_transpose --warm-up-time 1 --measurement-time 5 --sample-size 10
```

Same-worker `ts1`:

- Baseline `einsum/rhs_transpose_8x2048x2048`: `[65.985 ms 66.676 ms 67.123 ms]`
- After `einsum/rhs_transpose_8x2048x2048`: `[25.073 ms 25.325 ms 25.876 ms]`
- Post-rebase after `einsum/rhs_transpose_8x2048x2048`: `[25.011 ms 25.201 ms 25.367 ms]`

Final median speedup: `2.65x`.

Score: `Impact 4.0 x Confidence 0.95 / Effort 1.2 = 3.17`, keep.

## Isomorphism Proof

- Scope: no-grad f64 only. If either input requires grad, has a different dtype, has a non-contiguous layout, includes batch indices, has different index order, or needs a different output order, the code falls back to the existing `tensor_permute`/reshape/matmul path.
- Ordering: output shape/order remains `free_lhs` followed by `free_rhs`, matching the existing fast-path condition and final reshape order.
- Floating point: `dgemm_bt` is already tested in `ft-kernel-cpu` as bit-for-bit identical to materialize-transpose-then-`dgemm`; the new API test compares `tensor_einsum("ik,jk->ij")` against explicit `tensor_transpose` + `tensor_matmul` by `to_bits`.
- RNG: no RNG calls or seeds changed; the benchmark input generation remains the existing session `tensor_randn` setup.
- Autograd: requires-grad inputs are deliberately excluded; a focused test confirms `einsum_rhs_transpose_requires_grad_uses_autograd_path` still produces an input gradient.

## Validation

- `cargo check -p ft-kernel-cpu -p ft-api --all-targets` via `rch` on `ts1`: passed (`check_einsum_dgemm_bt.txt`).
- `cargo test -p ft-api einsum_rhs_transpose -- --nocapture` via `rch` on `ts1`: passed, `2 passed` (`test_einsum_rhs_transpose.txt`).
- Post-rebase `cargo check -p ft-kernel-cpu -p ft-api --all-targets` via `rch` on `ts1`: passed (`post_rebase_check_einsum_dgemm_bt.txt`).
- Post-rebase `cargo test -p ft-api einsum_rhs_transpose -- --nocapture` via `rch` on `ts1`: passed, `2 passed` (`post_rebase_test_einsum_rhs_transpose.txt`).
- Post-rebase `cargo bench -p ft-api --bench ops_bench -- einsum/rhs_transpose --warm-up-time 1 --measurement-time 5 --sample-size 10` via `rch` on `ts1`: passed (`post_rebase_after_rhs_transpose_dgemm_bt.txt`).
- `cargo clippy -p ft-kernel-cpu --all-targets --no-deps -- -D warnings` via `rch` on `ts1`: passed (`clippy_ft_kernel_cpu.txt`).
- `cargo clippy -p ft-api --lib --no-deps -- -D warnings` was attempted and remains blocked by the pre-existing `ft-api` lint backlog (`183` library errors), with no diagnostics for the new einsum fast path (`clippy_ft_api_lib.txt`).
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`: passed (`golden_sha256_check.txt`).
- `git diff --check`: passed (`git_diff_check.txt`).
- `ubs crates/ft-kernel-cpu/src/lib.rs crates/ft-api/src/lib.rs crates/ft-api/benches/ops_bench.rs artifacts/perf/frankentorch-isc4/report.md` was attempted with a 120 second timeout and exited `124` without actionable findings; partial log and status are retained in `ubs_changed_surface.txt` and `ubs_status.txt`.

## Next Primitive

Re-profile after landing. If einsum remains hot, the next deeper primitive is a broader no-grad strided contraction kernel covering LHS-transpose and batched cases, with explicit fallback for autograd and non-contiguous layouts.
