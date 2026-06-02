# Matmul Backward Parallel Gradient Cells Pass 14

Bead: `frankentorch-jr5u`

## Target

Profile-backed fallback target after `br ready --json` stayed empty:

- Fresh rch Criterion slice on worker `vmi1227854`.
- `backward_matmul/size/256` time: `[48.357 ms 50.725 ms 53.554 ms]`.
- `matmul/square/1024` time: `[51.040 ms 52.843 ms 54.742 ms]`.

The selected target was the normal `TensorNodeOp::MatMul` backward path. Forward matmul already routes through the `matrixmultiply`-backed kernel, while normal backward computes each gradient cell with explicit loops.

## Lever Attempted

One lever was attempted and then reverted:

- Add Rayon to `ft-autograd`.
- For large normal matmul backward outputs, compute independent lhs and rhs gradient cells with `par_chunks_mut`.
- Preserve each cell's inner accumulation order, output index, diagnostics, and RNG behavior.

## Behavior Proof

Focused rch checks before rejection:

```text
rch exec -- cargo check -p ft-autograd --all-targets
rch exec -- cargo test -p ft-autograd tensor_matmul_forward_backward_matches_expected_gradients -- --nocapture
rch exec -- cargo test -p ft-autograd tensor_matmul_backward -- --nocapture
```

All commands exited 0. A pass-local large deterministic test exercised the parallel branch before the code was reverted.

Isomorphism notes for the attempted lever:

- Ordering and tie-breaking: each gradient cell kept the original row/inner/column output index.
- Floating point: each cell kept the original inner accumulation order.
- RNG: not involved.
- Diagnostics: shape and length validation remained before the branch.

## Re-benchmark

After run on worker `vmi1153651`:

```text
backward_matmul/size/256 time: [44.829 ms 51.716 ms 61.043 ms]
```

The p50 moved 50.725 ms -> 51.716 ms against the fresh baseline, with one high severe outlier. This does not prove a real win.

## Verdict

Rejected and reverted. Score: impact -1 x confidence 2 / effort 2 = -1.0.
