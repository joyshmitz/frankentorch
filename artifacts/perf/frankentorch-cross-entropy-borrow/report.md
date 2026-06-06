# frankentorch-ib63 - ft-api cross_entropy borrowed logits

## Target

Profile-backed fallback target after `br ready --json` had no unclaimed ready perf bead. The fused f64 `functional_cross_entropy(..., "mean")` grad path saved the full `[batch, classes]` logits tensor with `ctx.save_for_backward(logits.to_vec(), vec![b, c])` before backward. At `4096x8192`, that is a full 33,554,432-element f64 copy per training step.

One lever: route only the f64 fused grad cross-entropy custom op through `tensor_apply_function_f64_borrowed_inputs` so backward reborrows the immutable input tensor values from the tape.

## Benchmark

Command:

```text
rch exec -- cargo bench -p ft-api --bench ops_bench -- cross_entropy/grad_4096x8192 --warm-up-time 1 --measurement-time 5 --sample-size 10
```

Same-path, same-worker `ts1` comparison:

| Run | Worker | Criterion interval |
| --- | --- | --- |
| Baseline | ts1 | [612.65 ms 618.58 ms 624.49 ms] |
| After confirm | ts1 | [542.14 ms 560.08 ms 582.05 ms] |

Additional patched same-path run:

| Run | Worker | Criterion interval |
| --- | --- | --- |
| After first | ts1 | [531.57 ms 539.49 ms 546.36 ms] |

Score:

```text
Impact 2.5 x Confidence 0.90 / Effort 1.0 = 2.25
```

Verdict: keep. Confirmation speedup is `618.58 / 560.08 = 1.105x` with non-overlapping Criterion intervals. The first patched run showed `1.146x`. The lever also removes the logits copy allocation from the backward path.

## Isomorphism Proof

Ordering:
- Target validation and `target_idx` construction order are unchanged.
- Per-row loss order and row-major gradient order are unchanged.
- Reduction dispatch after the per-row loss is unchanged.

Tie-breaking:
- No sorting, arg selection, or class tie-breaking logic changed.
- Class lookup continues to use the same target index vector.

Floating point:
- Forward still calls `ft_kernel_cpu::cross_entropy_forward_f64(logits, &ti_fwd, b, c)`.
- Backward still calls `ft_kernel_cpu::cross_entropy_backward_f64(logits, &ti_bwd, dloss, b, c)`.
- No arithmetic expression, loop order, accumulation order, log-sum-exp path, or softmax formula changed.
- Only the logits storage source changed from an owned saved copy to a borrowed immutable tape slice.

RNG:
- The operator contains no RNG.
- The benchmark setup still uses the same `tensor_randn` fixture construction outside the optimized custom-op backward lever.

Borrow contract:
- Backward receives immutable borrowed input slices from the tape.
- No mutable aliasing or in-place mutation path was introduced.
- This follows the same opt-in borrowed-input custom-autograd contract already used by the conv2d borrowed-input path.

Golden output:
- `cross_entropy_borrowed_grad_zero_logits_golden_bits` freezes the exact zero-logits mean-loss bits and gradient bits.
- Focused command passed 12 cross-entropy tests, including finite-difference and op-graph equivalence tests.

## Validation

Passed:
- `rch exec -- cargo test -p ft-api cross_entropy -- --nocapture`
- `rch exec -- cargo check -p ft-api --all-targets`
- `git diff --check`
- Post-rebase onto `origin/main` commit `2735f8d6`: `rch exec -- cargo test -p ft-api cross_entropy -- --nocapture`
- Post-rebase onto `origin/main` commit `2735f8d6`: `rch exec -- cargo check -p ft-api --all-targets`

Known unrelated gate backlog:
- `cargo clippy -p ft-api --all-targets -- -D warnings` fails on broad existing `ft-api` lint debt outside this change, including range-loop, excessive-precision constants, redundant closures, and unrelated recurrent-test `unused_mut`.
- `cargo fmt -p ft-api --check` fails on broad existing formatting drift in benches, examples, and unrelated regions of `crates/ft-api/src/lib.rs`.
- `ubs crates/ft-api/src/lib.rs` was run with a 120s cap; the captured log contains the scanner banner and rust-scan start only, with no finding lines.
- The commit hook's extended large-file UBS scan also timed out and instructed `UBS_SKIP=1 git commit ...`; that timeout is captured in `ubs_commit_hook_timeout.txt`.

Focused changed lines:
- Runtime lever: `crates/ft-api/src/lib.rs:9122`.
- Golden test: `crates/ft-api/src/lib.rs:71373`.

## Evidence Files

- `baseline_samepath_cross_entropy_grad.txt`
- `after_734_cross_entropy_grad.txt`
- `after_samepath_confirm_cross_entropy_grad.txt`
- `test_cross_entropy.txt`
- `post_rebase_test_cross_entropy.txt`
- `post_rebase_check_ft_api.txt`
- `check_ft_api.txt`
- `clippy_ft_api.txt`
- `fmt_ft_api.txt`
- `diff_check.txt`
- `ubs_ft_api.txt`
- `ubs_commit_hook_timeout.txt`
- `golden_cross_entropy_bits.txt`
- `source_diff.txt`
