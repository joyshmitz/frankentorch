# Conv2d Flattened Projection Pass 12

Bead: `frankentorch-hx4y`

## Target

Profile-backed fallback target after `br ready --json` stayed empty:

- Existing scenario bundle: `tests/artifacts/perf/20260601T2325Z-rustickite/SCENARIO.md` ranks `conv2d/hw/32` as a remaining high-gap row.
- Fresh rch Criterion baseline on worker `vmi1293453`: `conv2d/hw/32` time `[107.72 ms 110.37 ms 113.00 ms]`.
- `cargo flamegraph` was attempted through rch, but the worker denied profiling with `perf_event_paranoid=4`.

## Lever

One lever in `FrankenTorchSession::functional_conv2d`:

- Keep the existing patch extraction and transposed weight.
- Reshape unfolded patches from `[batch, patch_count, patch_width]` to `[batch * patch_count, patch_width]`.
- Use one shared `tensor_matmul` with the transposed weight.
- Reshape the result back to `[batch, patch_count, out_channels]`.

This avoids materializing an expanded per-batch weight tensor and avoids routing the projection through `tensor_bmm`.

## Behavior Proof

Focused rch tests:

```text
rch exec -- cargo check -p ft-api --all-targets
rch exec -- cargo test -p ft-api functional_conv2d_with_bias -- --nocapture
rch exec -- cargo test -p ft-api unfold -- --nocapture
rch exec -- cargo test -p ft-nn conv2d_backward_produces_gradients -- --nocapture
```

All commands exited 0.

`rch exec -- cargo clippy -p ft-api --all-targets -- -D warnings` exited 101 on existing ft-api warning/clippy backlog outside the changed conv2d projection block. `rch exec -- cargo fmt -p ft-api --check` exited 1 on existing file-wide rustfmt drift; the pass-local diff passed `git diff --check`.

`ubs crates/ft-api/src/lib.rs` exited 1 after scanning the single changed Rust file because the file already contains broad inventory findings, including panic/unwrap surfaces, direct indexing, and heuristic token-comparison false positives. No finding was reported for the changed conv2d projection block.

Golden sha256:

```text
d4b6fb888a9e0feec659f1201ec3227116a3ecb891f23e00d898a77ad7b01b6e  artifacts/optimization/golden_outputs/conv2d_flatten_projection_pass12.txt
```

`sha256sum -c artifacts/optimization/golden_checksums.txt` exited 0 after recording the pass-local fixture.

Isomorphism notes:

- Ordering and tie-breaking: patch extraction order, transposed weight layout, output transpose, and output reshape are unchanged.
- Floating point: each output element still uses the same dot-product order through the existing matmul kernel. Batch/patch grouping changes only the outer scheduling shape.
- Gradients: the focused `ft-nn` conv2d backward smoke remained unchanged.
- RNG: not involved.
- Diagnostics: shape checks and overflow checks remain fail-closed; a new checked multiplication guards `batch * patch_count`.

## Re-benchmark

Initial after run on worker `vmi1227854`:

```text
conv2d/hw/32 time: [132.43 ms 135.64 ms 138.97 ms]
```

Because the fresh baseline used another worker and a concurrent BMM pass changed the old path, a same-worker control was measured by temporarily restoring only the original conv2d projection:

```text
conv2d/hw/32 time: [131.74 ms 141.47 ms 154.79 ms]
```

Repeat flattened run on worker `vmi1149989`:

```text
conv2d/hw/32 time: [110.99 ms 114.37 ms 117.90 ms]
```

Same-worker control delta: 141.47 ms -> 135.64 ms, about 4.1 percent faster by p50. Confidence is intentionally low because the control had a high severe outlier and the tree had a concurrent BMM optimization affecting only the old `tensor_bmm` path.

## Verdict

Kept. Score: impact 1 x confidence 2 / effort 1 = 2.0.
