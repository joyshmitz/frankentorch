# frankentorch-789f ft-autograd MatMul Backward All-Ones Pass

## Target

- Bead: `frankentorch-789f`
- Scenario: `backward_matmul/size/128`
- Baseline command: `rch exec -- cargo bench -p ft-api --bench ops_bench -- backward_matmul/size/128 --warm-up-time 1 --measurement-time 5 --sample-size 20`
- Baseline worker/result: `vmi1293453`, `[5.4378 ms 5.5910 ms 5.7065 ms]`
- Profile note: the same harness measured forward-only `matmul/square/128` at `[229.32 us 277.41 us 327.21 us]`, making the backward/autograd path the profiler-evident target.

## Lever

Specialize `TensorNodeOp::MatMul` backward when the incoming gradient is exactly the all-ones tensor produced by `sum(matmul(...))`. The old code recomputed identical row and column reductions for every output row/column. The new path detects exact `1.0_f64` bits and computes each repeated reduction once, then fills the repeated contribution slots.

## Isomorphism Proof

- Ordering preserved: the non-all-ones path is byte-for-byte the old loop order; the all-ones path keeps the same inner reduction order for each unique contribution and only avoids recomputing identical reductions.
- Tie-breaking unchanged: no comparator, sorting, or selection behavior is touched.
- Floating-point preserved: the gate is an exact bit check for positive `1.0`; accumulated products are still `1.0 * value` in the old left-to-right order for each unique sum.
- RNG unchanged: no random state, seeds, or sampler behavior involved.
- DAC evidence preserved: the golden fixture records forward values, scalar loss, lhs/rhs gradients, step count, and execution order.
- Golden output sha256: `b106ad03f779629f712d72e594a0c3733afbe55740ea401c85680a604b8c8efa`.

## Verification

- `rch exec -- cargo test -p ft-autograd tensor_matmul -- --nocapture`: passed, 3 tests.
- `rch exec -- cargo check -p ft-autograd --all-targets`: passed.
- `rch exec -- cargo clippy -p ft-autograd --all-targets --no-deps -- -D warnings`: passed.
- `rch exec -- cargo fmt -p ft-autograd --check`: passed.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`: passed.
- `ubs --only=rust crates/ft-autograd/src/lib.rs`: exit 1 from pre-existing broad-file findings. The critical examples were existing test panic macros and false-positive non-secret shape comparisons; the new matmul specialization and golden fixture were not listed as critical findings.

## Benchmark Results

- Before: `[5.4378 ms 5.5910 ms 5.7065 ms]`.
- After: `[1.3279 ms 1.3989 ms 1.4704 ms]`.
- Median delta: `5.5910 ms -> 1.3989 ms`, about `4.0x` faster.
- Throughput delta: `[2.8711 Melem/s 2.9304 Melem/s 3.0130 Melem/s] -> [11.142 Melem/s 11.712 Melem/s 12.338 Melem/s]`.

## Decision

Kept. Impact `4` x confidence `2` / effort `1` = `8.0`, above the required `2.0` threshold. Confidence is capped because the after-run landed on a different RCH worker, but the effect size is large enough to keep.
