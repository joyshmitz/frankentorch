# frankentorch-kgs4.119 - conv3d f64 borrowed-input autograd

Agent: OrangeCedar / cod-b
Date: 2026-06-18
Status: in_progress, code-first batch-test pending

## Workload Trigger

Current realistic train reprofile shows `conv3d/grad` in the first tier of
remaining wall time:

- `artifacts/perf/frankentorch-next-reprofile-20260617/current_top_train_reprofile.log`
- Baseline row: `conv3d/grad [40.824 ms 41.312 ms 41.921 ms]`
- Later routing rows after adjacent train work: about `38 ms` median in
  `20260617b` and `20260617c`.

## Lever

The f64 `functional_conv3d` grad fast path still saved full owned padded-input
and weight copies in `FunctionCtx`:

- `ctx.save_for_backward(pv.to_vec(), ...)`
- `ctx.save_for_backward(wv.to_vec(), ...)`

The underlying forward, backward, and create-graph adjoint kernels already exist.
This patch routes the f64 conv3d custom op through
`apply_function_with_create_graph_borrowed_inputs`, mirroring the proven f64
conv2d path, so first-order backward reads immutable input slices from the tape
instead of cloning full tensors into the context.

## Correctness Guard

Added `conv3d_f64_borrowed_input_grad_contract` beside the existing finite-diff
conv3d gradient test. It checks a small f64 conv3d backward contract:

- input gradient equals the 1x1x1 weight for `loss=sum(out)`
- weight gradient equals the sum of input values
- the existing `conv3d_grad_matches_finite_diff` remains the broader dinput,
  dweight, dbias, stride/padding guard.

No math kernel, accumulation order, dtype, shape, or create-graph path was
changed.

## Negative-Evidence Ledger

Do not confuse this with these already-recorded dead ends:

- `frankentorch-b03fn`: f64 max_pool2d borrowed-input-only path regressed
  `99.832 ms -> 108.28 ms`; do not retry pool borrowed-input tape plumbing.
- `frankentorch-cjfsb`: f32 linear no-grad borrowed-input path rejected; GEMM
  dominated and clone removal was too small.
- `frankentorch-xbvlx`: scalar 2x2 max_pool2d direct backward rejected; same
  worker candidate regressed the measured rows.
- `frankentorch-2rsa6`: group_norm saved-stats rematerialization rejected.
- `frankentorch-kgs4.110`: BatchNorm spatial1 row-major f64 reduction rejected.

Positive precedent:

- `frankentorch-6uw9`: f64 conv2d borrowed-input custom autograd was productive,
  same-worker `conv2d/grad_hw/64` median `223.02 ms -> 177.43 ms`.

## Verification

Requested local-only gate for this code-first pass:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b cargo check -p ft-api
```

Result: PASS on 2026-06-18.

Criterion/conformance batch testing is intentionally pending per campaign
instruction.
