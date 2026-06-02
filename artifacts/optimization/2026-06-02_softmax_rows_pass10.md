# Softmax Row Parallelization Pass 10

Bead: `frankentorch-ani4`

## Target

Profile-backed fallback target after pass 9:

- 2026-06-01 scenario: `softmax/vocab/8192` measured [1.5091 ms 1.5120 ms 1.5157 ms].
- Fresh rch baseline: `softmax/vocab/8192` measured [1.8068 ms 1.8415 ms 1.8737 ms] on worker `vmi1293453`.

The active conv2d/unfold work owned `ft-api`, so this pass stayed in
`ft-kernel-cpu`.

## Lever

One lever in `softmax_dim_tensor_contiguous_f64`:

- For `inner_size == 1`, process independent contiguous rows with
  `par_chunks_mut(reduce_size)` zipped with input row chunks.
- Inside each row, preserve the original max scan, exp writes, pairwise sum, and
  normalization loops.

## Behavior Proof

Focused rch tests:

```text
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-ani4-target rch exec -- cargo test -p ft-kernel-cpu softmax_dim_sums_to_one -- --nocapture
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-ani4-target rch exec -- cargo test -p ft-kernel-cpu softmax_dim_pairwise_sums_exactly_to_one_at_large_n -- --nocapture
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-ani4-target rch exec -- cargo check -p ft-kernel-cpu --all-targets
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-ani4-target rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings
```

All commands exited 0.

Golden sha256:

```text
4598d235c3ce9cd4d913bdecaab1d60a9ae2d67f1607eeb6f7a552758e74c5fd  artifacts/optimization/golden_outputs/softmax_rows_pass10.txt
```

`sha256sum -c artifacts/optimization/golden_checksums.txt` exited 0.

Isomorphism notes:

- Ordering and tie-breaking: row order and element order within each row are unchanged.
- Floating point: each row keeps the same max scan, exp write order, pairwise sum order,
  and normalization order. Only independent rows are scheduled in parallel.
- NaN/Inf behavior: per-row max subtraction and normalization are unchanged.
- RNG: not involved.
- Diagnostics: validation and dimension handling are unchanged.

## Re-benchmark

Initial after run on worker `ts2`:

```text
softmax/vocab/8192 time: [1.0754 ms 1.1061 ms 1.1531 ms]
```

Because the fresh baseline used another worker, a same-worker serial control was
measured by temporarily restoring the original row loop:

```text
softmax/vocab/8192 time: [3.6487 ms 3.6761 ms 3.7035 ms]
```

Same-worker delta: 3.6761 ms -> 1.1061 ms, about 69.9 percent faster.

## Formatting

`rch exec -- cargo fmt -p ft-kernel-cpu --check` exited 1 due to the existing
file-wide rustfmt drift in `crates/ft-kernel-cpu/src/lib.rs`. The pass-local diff
passed `git diff --check`.

## Verdict

Kept. Score: impact 3 x confidence 3 / effort 2 = 4.5.
