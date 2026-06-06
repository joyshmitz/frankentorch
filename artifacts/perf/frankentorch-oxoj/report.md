# frankentorch-oxoj: batched RHS-transpose einsum dgemm_bt fast path

## Target

- Bead: `frankentorch-oxoj`
- Profile-backed target: follow-up to `frankentorch-isc4`, where no-grad f64 RHS-transpose einsum improved `66.676 ms -> 25.201 ms` on `ts1`.
- New benchmark: `einsum/batched_rhs_transpose_16x8x512x512`, equation `bik,bjk->bij`.

## Lever

Add one no-grad f64 fast path in `einsum_binary` for contiguous positive-size batched contractions shaped:

- LHS: `batch, free_lhs, contract`
- RHS: `batch, free_rhs, contract`
- Output: `batch, free_lhs, free_rhs`

The fast path calls `ft_kernel_cpu::matmul_rhs_transposed_contiguous_f64` per batch, so it reads RHS as `[free_rhs, contract]` without materializing `tensor_permute(rhs, [batch, contract, free_rhs])`.

All requires-grad, non-f64, non-contiguous, reordered, broadcasted, and zero-size cases fall back to the existing general einsum path.

## Benchmark

Same worker: `ts1`

- Baseline command: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- einsum/batched_rhs_transpose --warm-up-time 1 --measurement-time 5 --sample-size 10`
- Baseline: `[81.817 ms 90.124 ms 98.537 ms]`
- After, post-rebase confirmation: `[28.976 ms 30.668 ms 31.511 ms]`
- Median speedup: `90.124 / 30.668 = 2.94x`
- Score: `Impact 2.94 x Confidence 0.95 / Effort 1.1 = 2.54`
- Verdict: keep.

The pre-rebase after run was `[25.690 ms 25.838 ms 26.070 ms]`; the post-rebase run is the final keep gate.

## Isomorphism

- Ordering: `par_chunks_mut(...).enumerate()` writes disjoint output batch chunks by batch index; collection order does not affect values or output order.
- Tie-breaking: no comparisons or ties are introduced.
- Floating point: for each batch, `dgemm_bt` reads RHS rows through matrixmultiply strides matching materialized transpose plus `dgemm`, preserving the per-output K accumulation order. The focused test compares every output with `to_bits`.
- RNG: runtime fast path uses no RNG. Benchmark tensors are allocated once before Criterion iteration, matching the baseline benchmark lifecycle.
- Autograd: requires-grad inputs bypass the fast path; focused gradient test proves the existing autograd route still records and returns the expected LHS gradient.

## Evidence

- Compile: `check_post_rebase.txt` passed.
- Focused tests: `test_post_rebase.txt` passed.
- Golden output: `golden_output_batched_rhs_transpose.txt`, with explicit output and gradient values; included in `evidence.sha256`.
- Diff hygiene: `git diff --check` passed for the final worktree diff.
- Clippy: `clippy_ft_api_final.txt` still fails on the known `ft-api` backlog (`183` lib and `206` lib-test errors); no diagnostics hit the changed fast path or new tests.
- Format: final `cargo fmt --check` still fails on unrelated repo-wide rustfmt backlog; grep confirmed no remaining formatter diff at the changed batched-einsum lines.
- UBS: touched-file UBS timed out at 120s while scanning the large touched source file.

## Next Primitive

Re-profile ready perf beads. If einsum remains the best isolated target, attack profile-backed strided/broadcasted contractions with a strict fallback and bit-exact ledger; otherwise route to the next independent batched-LU/workspace primitive.
