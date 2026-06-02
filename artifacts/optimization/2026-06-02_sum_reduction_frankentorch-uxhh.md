# Sum Reduction Pass: frankentorch-uxhh

## Target

- Bead: `frankentorch-uxhh`
- Skill loop: `/repeatedly-apply-skill` applying `/extreme-software-optimization`
- Crate path: `ft-kernel-cpu` sum reduction, measured through the existing `ft-api` Criterion bench `sum/elements/1000000`
- Candidate hotspot: `sum_tensor_contiguous_f64` -> `pairwise_sum_f64`

## Baseline

Command:

```text
rch exec -- cargo bench -p ft-api --bench ops_bench -- sum/elements/1000000 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Result on worker `vmi1149989`:

```text
sum/elements/1000000 time [274.21 us 279.48 us 288.88 us]
throughput [3.4616 Gelem/s 3.5781 Gelem/s 3.6469 Gelem/s]
```

Profiler note:

```text
rch exec -- perf record -F 99 --call-graph dwarf -o /tmp/frankentorch-uxhh-sum.perf -- cargo bench -p ft-api --bench ops_bench -- sum/elements/1000000 --profile-time 3
```

`perf` was blocked by `perf_event_paranoid=4`, so no call graph was available. The target remained the dedicated Criterion reduction hotspot rather than an unprofiled source guess.

## Lever Attempted

Attempted one source lever:

- Replace the leaf `.iter().sum()` in `pairwise_sum_f64` with an explicit left-to-right loop.
- Keep the `BLOCK = 128` threshold unchanged.
- Keep recursive split points unchanged.

Isomorphism obligations:

- Pairwise split tree unchanged.
- Leaf floating-point addition order unchanged.
- Storage-offset slicing unchanged.
- Empty input remains `0.0`.
- Ordering, tie-breaking, and RNG are unaffected.

## Behavior Proof

Golden output:

```text
5f0fcdf7cf26d8b679cc7aad126c834fe469779595351aca18c46f4f7ad4374a  artifacts/optimization/golden_outputs/sum_reduction_frankentorch-uxhh.txt
```

Verification:

```text
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
rch exec -- cargo test -p ft-kernel-cpu sum_tensor_contiguous -- --nocapture
```

Focused test result:

```text
running 6 tests
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 363 filtered out
```

The new executable golden test pins exact f64 bit patterns for empty, storage-offset, boundary, and 1,048,576-element uniform cases.

## Re-benchmark

Attempted lever result:

```text
sum/elements/1000000 time [1.6413 ms 1.7489 ms 1.8928 ms]
throughput [528.32 Melem/s 571.80 Melem/s 609.26 Melem/s]
```

Decision: rejected. The candidate regressed by about 6.26x by mean (`279.48 us` -> `1.7489 ms`), so the production hot-path change was reverted.

Restored-code confirmation:

```text
sum/elements/1000000 time [274.60 us 278.78 us 285.73 us]
throughput [3.4999 Gelem/s 3.5870 Gelem/s 3.6417 Gelem/s]
```

The retained diff is performance-neutral and keeps only the golden proof plus negative-result evidence.

## Gates

```text
cargo fmt -p ft-kernel-cpu --check
ubs crates/ft-kernel-cpu/src/lib.rs artifacts/optimization/golden_checksums.txt artifacts/optimization/golden_outputs/sum_reduction_frankentorch-uxhh.txt
rch exec -- cargo test -p ft-kernel-cpu sum_tensor_contiguous -- --nocapture
rch exec -- cargo check -p ft-kernel-cpu --all-targets
rch exec -- cargo clippy -p ft-kernel-cpu --all-targets --no-deps -- -D warnings
```

All gates passed. UBS reported existing broad warnings in `ft-kernel-cpu`, no critical issues, and no new perf/DoS allocation warning from the added golden fixture.

## Score

- Impact: -2
- Confidence: 3
- Effort: 1
- Score: -6.0

Status: no optimization kept; close as negative-result profile evidence.
