# frankentorch-t1vg pass 1 baseline/profile

Date: 2026-06-07
Scope: read/profile only. No intentional source edits in this pass.
Skill: extreme-software-optimization

## Bead

- `frankentorch-t1vg`: `[perf][no-gaps] ft-api: borrow Linear grad inputs in custom autograd`
- Status observed with `br show frankentorch-t1vg --json`: `in_progress`
- Assignee observed: `RubyLotus`
- `br ready --json`: `[]`

## Benchmark Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- linear_train/hidden/2048 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

RCH selected worker:

- `vmi1227854` at `ubuntu@109.123.245.77`
- RCH trailer: `[RCH] remote vmi1227854 (142.0s)`

Criterion result:

```text
linear_train/hidden/2048
time: [45.873 ms 46.683 ms 47.530 ms]
```

Benchmark source:

- `crates/ft-api/benches/ops_bench.rs:955`: `bench_linear_train`
- Dimensions for `hidden/2048`: `batch = 32`, `in_features = 512`, `hidden = 2048`
- Per iteration: new `FrankenTorchSession`, random `x`, `w`, `bias` all with `requires_grad = true`, then `tensor_linear`, `tensor_sum`, and `tensor_backward`.

Important source-state note:

- The baseline command was started from the initial pass state where `git status --short` did not show `crates/ft-api/src/lib.rs` modified, and an initial source grep found the owned `ctx.save_for_backward(x.to_vec(), xs.to_vec())` / `ctx.save_for_backward(w.to_vec(), ws.to_vec())` target in `functional_linear`.
- During/after the baseline run, the shared working tree acquired an uncommitted borrowed-input hunk in `crates/ft-api/src/lib.rs`. I did not edit that file. Current `git diff -- crates/ft-api/src/lib.rs` shows the intended candidate lever already present as an uncommitted change.

## Source Locations

Current Linear f64 grad fast path:

- `crates/ft-api/src/lib.rs:16046`: `functional_linear`
- `crates/ft-api/src/lib.rs:16113`: f64 grad fast-path comment
- `crates/ft-api/src/lib.rs:16147`: current working tree calls `tensor_apply_function_f64_borrowed_inputs`
- `crates/ft-api/src/lib.rs:16160`: backward closure reads `borrowed_inputs`

The current uncommitted hunk replaces the owned custom-op path:

```diff
-                    return self.tensor_apply_function(
+                    return self.tensor_apply_function_f64_borrowed_inputs(
...
-                            ctx.save_for_backward(x.to_vec(), xs.to_vec());
-                            ctx.save_for_backward(w.to_vec(), ws.to_vec());
...
-                        move |ctx, grad_outputs| {
+                        move |_ctx, grad_outputs, borrowed_inputs| {
...
-                            let saved = ctx.saved_tensors();
+                            let x = borrowed_inputs[0].0;
+                            let w = borrowed_inputs[1].0;
```

Borrowed-input helper:

- `crates/ft-api/src/lib.rs:6428`: `tensor_apply_function_f64_borrowed_inputs` API helper
- `crates/ft-api/src/lib.rs:6450`: delegates to `self.tensor_tape.apply_function_f64_borrowed_inputs`
- `crates/ft-autograd/src/lib.rs:8392`: tape implementation reads immutable f64 input slices during forward
- `crates/ft-autograd/src/lib.rs:8457`: stores `CustomFunctionBackward::BorrowedInputsF64`
- `crates/ft-autograd/src/lib.rs:13526`: backward reconstructs borrowed input slices from tape values

Owned save-for-backward context:

- `crates/ft-autograd/src/lib.rs:1411`: `FunctionCtx` stores `saved_tensors: Vec<Vec<f64>>`
- `crates/ft-autograd/src/lib.rs:1427`: `save_for_backward` pushes owned value and shape vectors
- `crates/ft-autograd/src/lib.rs:1434`: `saved_tensors()` returns the owned saved vectors

## Cost Evidence

For `linear_train/hidden/2048`:

- `x`: `32 * 512 = 16,384` f64 values = `128 KiB`
- `w`: `2048 * 512 = 1,048,576` f64 values = `8 MiB`
- Old owned context materialization cloned about `1,064,960` f64 values per forward = `8.125 MiB`, plus two small shape vectors.
- Bias is not saved in the old target path; `db` is derived from `dy`.
- Linear math work is about `32 * 512 * 2048 = 33,554,432` multiply-adds for forward, another same-sized pass for `dx`, and another for `dw`, roughly `100.7M` multiply-adds per iteration plus `65,536` bias-gradient additions.

No separate `perf` sample was captured in this pass. The profile evidence is the same-worker Criterion baseline plus the source-level allocation/copy cost model above.

## Proof Status

Golden checksum command:

```bash
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
```

Result: all locally present golden outputs reported `OK`.

Focused test command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-api linear -- --nocapture
```

First attempt: failed before running because no admissible remote worker was available while remote execution was required.

Retry result:

- Worker: `vmi1227854`
- RCH trailer: `[RCH] remote vmi1227854 (86.5s)`
- Result: `23 passed; 0 failed; 0 ignored; 0 measured; 1903 filtered out`
- Included `tests::functional_linear_f64_grad_fused_matches_analytic`
- Also included `tests::functional_linear_f32_fused_matches_transpose_path_bit_exact`

Warnings observed on the current working tree:

- `crates/ft-api/src/lib.rs:16151`: borrowed-input hunk leaves `ws` unused.
- Three existing `unused_mut` warnings in `crates/ft-api/src/lib.rs` test closures.

The `ws` warning is candidate-hunk-local and must be fixed before any clippy `-D warnings` proof for a keep commit.

## Files Changed

Intentional changes from this pass:

- `artifacts/perf/frankentorch-t1vg/pass1_baseline_profile.md`

Observed concurrent/pre-existing working-tree changes not edited by this pass:

- `.beads/issues.jsonl`
- `.skill-loop-progress.md`
- `crates/ft-api/src/lib.rs`
- `artifacts/perf/frankentorch-57nj/`

No files were deleted.
