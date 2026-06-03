# ft-api Conv2d Fused Panel

- Bead: `frankentorch-a29q`
- Target benchmark: `ops_bench/conv2d/hw/{32,64,128}`
- Lever: no-grad `functional_conv2d` fast path that gathers the im2col panel
  directly in patch-major order, replacing the unfold/unfold/permute/reshape
  materialization chain.

## Baseline

Clean committed-HEAD worktree, rch Criterion on `ts2`:

```text
conv2d/hw/32   time: [183.59 ms 185.32 ms 187.03 ms]
conv2d/hw/64   time: [901.83 ms 930.04 ms 968.48 ms]
conv2d/hw/128  time: [3.8540 s 3.8695 s 3.8851 s]
```

The profile target is memory-bound: the composed path materializes an unfolded
tensor, then materializes a second permuted copy before the same flattened
matmul. For hw=32 this is about a 19 MB panel plus another 19 MB permutation.

## Alien Primitive Mapping

- Graveyard primitive: nested data parallel flattening. The convolution patch
  grid is flattened directly into the GEMM panel shape instead of constructing
  nested intermediate tensors and then permuting them.
- Graveyard primitive: packed-panel GEMM layout. The fast path writes the
  `[batch * out_h * out_w, in_ch * kh * kw]` panel in the exact order consumed
  by the existing matmul.

## Isomorphism Proof

- Ordering preserved: yes. Output remains NCHW; panel rows are patch-major
  `(batch, out_h, out_w)` and columns are `(channel, kernel_h, kernel_w)`, the
  same logical order produced by the composed path after permutation.
- Tie-breaking unchanged: yes. Conv2d has no tie-breaking path.
- Floating-point unchanged: yes. The matmul call, weight flattening, bias add,
  and accumulation order are unchanged; only the panel materialization path is
  fused for no-grad tensors.
- RNG unchanged: yes. The operator has no RNG path.
- Autograd unchanged: yes. Any input, weight, or bias requiring grad falls back
  to the existing composed path; the proof test compares fast no-grad output to
  composed output and keeps composed backward gradient digest fixed.

## Validation

- Proof test on `ts2`:
  `cargo test -p ft-api functional_conv2d_no_grad_fast_path_matches_composed_path_bit_exact -- --nocapture`
  passed.
- Crate check on `ts2`:
  `cargo check -p ft-api --all-targets` passed with pre-existing warnings.
- Golden output: `ft_api_conv2d_fused_panel_frankentorch-a29q.txt`, sha256
  `fb414ed3ce5db49ddfdd26e37e647754580f18884d9417333bc4cd0468859445`.
- `cargo fmt --check -p ft-api` and `cargo clippy -p ft-api --all-targets
  --no-deps -- -D warnings` remain blocked by pre-existing repository debt.
  The new conv2d hunk was manually matched to rustfmt output and
  `git diff --check` passed.
- `ubs --only=rust crates/ft-api/src/lib.rs` completed in 353s and reported the
  broad pre-existing `ft-api` inventory: 351 critical, 17893 warnings, 2195
  info. Its shadow-workspace checks reported formatting, clippy, cargo check,
  and test build clean; no new conv2d-specific UBS defect was isolated.

## Result

Same-worker candidate rebenchmark on `ts2`:

```text
conv2d/hw/32   time: [44.774 ms 45.176 ms 45.837 ms]
conv2d/hw/64   time: [139.65 ms 140.16 ms 140.62 ms]
conv2d/hw/128  time: [604.93 ms 607.39 ms 610.77 ms]
```

Delta by p50:

- `hw/32`: `185.32 ms -> 45.176 ms`, about `4.10x` faster.
- `hw/64`: `930.04 ms -> 140.16 ms`, about `6.64x` faster.
- `hw/128`: `3.8695 s -> 607.39 ms`, about `6.37x` faster.

Score: Impact 5 x Confidence 4 / Effort 2 = 10.0.

## Next Primitive After Re-profile

Re-profile after this commit. A separate uncommitted ft-kernel-cpu GEMM
crossover change exists in the shared tree and must be measured as its own
one-lever pass before it can be kept.
