# Elementwise SIMD Optimization Pass 7

Bead: `frankentorch-p8h0`

## Target

Profile-backed fallback target from the 2026-06-01 measurement bundle:

- `relu/elements/1000000`: [606.47 us 611.74 us 620.65 us] in the prior local scenario.
- `add/elements/1000000`: [783.44 us 786.81 us 792.39 us] in the prior local scenario.

The scenario classed `relu` and `add` as large PyTorch-gap elementwise rows.
`conv2d` was already owned by an active `ft-api` pass, so this pass stayed in
`ft-kernel-cpu`.

## Fresh RCH Baseline

Worker: `ts2`

Commands:

```text
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-p8h0-target rch exec -- cargo bench -p ft-api --bench ops_bench -- relu/elements/1000000 --warm-up-time 1 --measurement-time 5 --sample-size 20
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-p8h0-target rch exec -- cargo bench -p ft-api --bench ops_bench -- add/elements/1000000 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Results:

```text
relu/elements/1000000 time: [6.2332 ms 6.2710 ms 6.3010 ms]
add/elements/1000000  time: [6.0716 ms 6.4111 ms 6.8163 ms]
```

The rch timings are much slower than the prior local bundle, but the before and
after measurements for this pass were kept on the same worker with the same
benchmark filters.

## Lever

One lever in `ft-kernel-cpu`: change the f64 shared SIMD unary/binary helpers
from capacity growth via `extend_from_slice`/`push` to pre-sized output buffers
with chunk writes and tail writes.

No `unsafe` was added.

## Behavior Proof

Focused rch tests:

```text
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-p8h0-target rch exec -- cargo test -p ft-kernel-cpu relu_tensor_contiguous_returns_expected_values -- --nocapture
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-p8h0-target rch exec -- cargo test -p ft-kernel-cpu add_tensor_contiguous_returns_expected_values -- --nocapture
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-p8h0-target rch exec -- cargo test -p ft-kernel-cpu preserves_simd_tail_values -- --nocapture
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-p8h0-target rch exec -- cargo check -p ft-kernel-cpu --all-targets
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-p8h0-target rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings
```

All commands exited 0.

Golden output:

```text
relu_tensor_contiguous_f64 values: [0.0, 0.0, 2.0, 0.0]
add_tensor_contiguous_f64 values: [1.5, 3.5, 5.5, 7.5]
simd_tail_relu_f64 values: [0.0, 1.0, 2.0, 3.0, 0.0]
simd_tail_add_f64 values: [11.0, 22.0, 33.0, 44.0, 55.0]
```

Golden sha256:

```text
16085f7bfb42daeb9cc84f4fcd33fdf001b3331a8023507af6db3705db51f8c8  artifacts/optimization/golden_outputs/elementwise_simd_pass7.txt
```

`sha256sum -c artifacts/optimization/golden_checksums.txt` exited 0.

Isomorphism notes:

- Ordering and tie-breaking: element index order and output length unchanged.
- Floating point: each lane still applies the same scalar or SIMD operation once.
- Tail behavior: explicit length-5 tests cover scalar tail writes after one SIMD chunk.
- Storage offsets and diagnostics: metadata validation and slicing remain before helper entry.
- RNG: not involved.

## Re-benchmark

Worker: `ts2`

Commands:

```text
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-p8h0-target rch exec -- cargo bench -p ft-api --bench ops_bench -- relu/elements/1000000 --warm-up-time 1 --measurement-time 5 --sample-size 20
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-p8h0-target rch exec -- cargo bench -p ft-api --bench ops_bench -- add/elements/1000000 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Results:

```text
relu/elements/1000000 time: [5.7603 ms 5.8362 ms 5.9095 ms]
add/elements/1000000  time: [5.3918 ms 5.6735 ms 6.0246 ms]
```

Mean deltas:

- `relu/elements/1000000`: 6.2710 ms -> 5.8362 ms, about 6.9 percent faster.
- `add/elements/1000000`: 6.4111 ms -> 5.6735 ms, about 11.5 percent faster.

## Formatting

`rch exec -- cargo fmt -p ft-kernel-cpu --check` classified as a non-compilation
command and exited 1 because of broad pre-existing rustfmt drift in
`crates/ft-kernel-cpu/src/lib.rs`. The formatter diff included many unrelated
regions outside this pass. The pass-local diff passed `git diff --check`.

## Verdict

Kept. Score: impact 2 x confidence 2 / effort 1 = 4.0.
