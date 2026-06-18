# frankentorch-kgs4.112 code-first avg_pool2d backward 2x2s2 specialization

- Bead: `frankentorch-kgs4.112`
- Agent: `cod-b` / Agent Mail `IvoryDeer`
- Status: code-first commit candidate; bead intentionally left `in_progress`.
- Baseline source: `artifacts/perf/frankentorch-next-reprofile-20260617/current_top_train_reprofile.log`
- Hot row: `avg_pool2d/grad` for `[N,C,H,W]=[8,64,64,64]`, `2x2 stride2`, median `107.13 ms`.
- Baseline comparator: generic `avg_pool2d_backward_f64` window geometry loop.

## Lever

Specialize `avg_pool2d_backward_f64` when all of these hold:

- `kh == kw == sh == sw == 2`
- `pad_h == pad_w == 0`
- `count_include_pad == true`
- `ph == ih`, `pw == iw`
- output shape matches floor `2x2 stride2` pooling

The fast path writes the four scatter targets directly with `dout / 4.0`.

## Alien mapping

- Graveyard primitive: numeric-kernel cache/locality specialization, affine-loop specialization, vectorized/morsel-style execution.
- Artifact-coding family: certified rewrite pipeline for a linear operator.
- Proof obligation: preserve the linear map `A^T * dout` exactly for the guarded shape.
- EV sketch: Impact 2, Confidence 4, Effort 1 -> Score 8.0 before benchmark; actual keep decision is pending Criterion.

## Behavior preservation

- Ordering preserved: yes. Per plane, output traversal remains `oy` then `ox`; each output writes top-left, top-right, bottom-left, bottom-right.
- Floating-point preserved: yes. Each output gradient computes the same `dout[o] / 4.0`; no reductions are reassociated.
- Overlap behavior: guarded shape has non-overlapping windows, so direct writes are equivalent to the generic `+=` scatter from zero.
- Shape/error behavior: unchanged; guard runs after existing shape validation and only inside the existing backward helper.

## Guard

Added `avg_pool2d_2x2s2_backward_matches_generic_bit_exact` in `crates/ft-kernel-cpu/src/lib.rs`.
The test computes a hand-written generic reference using the old loop order and compares every `f64::to_bits()`.

## Verification

Command run per campaign instruction:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b cargo check -p ft-kernel-cpu
```

Result: passed locally on 2026-06-18.

Not run by instruction: tests, Criterion, `rch`, clippy, fmt.

## Negative-evidence ledger

Attempt: `avg_pool2d_backward_f64` 2x2 stride2 no-pad specialization.

Status: `PENDING_BATCH_BENCH`.

Benchmark verdict: not measured in this commit because the campaign instruction allowed only local `cargo check -p ft-kernel-cpu` and explicitly forbade tests/rch for this code-first batch.

Retry condition: run `cargo bench -p ft-api --bench ops_bench -- avg_pool2d/grad` in the batch-test window on a comparable target dir; keep only if the focused row improves beyond noise without regressing the broad training profile.

If rejected: do not retry 2x2 stride2 avg-pool backward shape-branching unless a later profile shows this exact helper remains a top-5 frame and the failed batch report identifies branch overhead, not memory bandwidth, as the miss reason.
