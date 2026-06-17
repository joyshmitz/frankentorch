# frankentorch-6olvt closeout

Decision: KEEP.

Profile-backed target:
- Source: `artifacts/perf/frankentorch-next-reprofile-20260617/current_top_train_reprofile.log`
- Row: `batch_norm/grad_train_32x256x28x28` `[503.95 ms 513.09 ms 520.41 ms]`

Dedicated local baseline:
- Command: `RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/tmp/frankentorch-6olvt-baseline-target cargo bench -j 1 -p ft-api --bench ops_bench -- 'batch_norm/grad_train_32x256x28x28' --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot`
- Result: `[527.39 ms 537.73 ms 548.59 ms]`
- Artifact SHA-256: `e70ff72aa65dab3d965c8c4b770c89a1eebf14a64984cdb82a0f96145468a892`

One lever:
- Added a guarded f64 BatchNorm backward path for exact all-ones upstream gradients (`dy == 1.0`) when `spatial != 1`.
- This targets the common `sum(batch_norm(...))` backward shape represented by the benchmark, avoiding repeated `dy` loads and `dy * weight` multiplications while preserving the existing general path for all non-one gradients.

Behavior proof:
- The guard is bit-exact only for `dy` elements whose bits equal `1.0f64`; all other `dy` inputs fall back to the previous general implementation.
- The specialized path preserves the old per-channel `n -> spatial` accumulation order for `dweight`, exact `dbias = M` (same as adding `1.0` `M` times for these sizes), dx formula order, affine math, dtype/error behavior, and RNG absence.
- Kernel proof compares the specialized path against a local copy of the old general reduction order.
- Golden digest: `0x4edb3f2ac54649ea`.
- Kernel proof SHA-256: `97127d65d2a9f7ac4a086a7be74291f139271ca60ec330203c90405506dbedf3`
- API proof: `functional_batch_norm2d_grad_matches_finite_diff` passed.
- API proof SHA-256: `914aa0cd6a4371b385c6bc0ede31c606ec530ec655ea8eabfdaf68cb095f7dbd`

Rebench:
- Candidate result: `[499.20 ms 507.76 ms 516.66 ms]`
- Criterion delta: `[-8.0994% -5.5726% -3.0796%]`, `p = 0.00`; performance improved.
- Baseline median to candidate median: `537.73 ms -> 507.76 ms`, `1.059x`.
- Rebench SHA-256: `27b192f1dfeacd90fb4334f89d411f9afcb395d2a69b3bad285b9feadfe8676e`

Score:
- `2.05 = Impact 1.059 * Confidence 0.97 / Effort 0.50`
- Above the `2.0` keep threshold.

Gates:
- `cargo check -j 1 -p ft-kernel-cpu --all-targets` passed with pre-existing example warnings; SHA-256 `17fa8467fed0b87740b76ab0e7ff501948cd06913dc15c7a05d24f82af9a9e3b`.
- `cargo clippy -j 1 -p ft-kernel-cpu --lib -- -D warnings` passed; SHA-256 `77eea0a8ba631007934638d23209772152dc828d80e4fef3bb61d2e1385c9d9d`.
- `git diff --check` passed; SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `rustfmt --edition 2024 --check crates/ft-kernel-cpu/src/lib.rs` reports pre-existing broad file drift outside this hunk; not applied to avoid unrelated churn. Artifact SHA-256 `69d50b1edb906ea8ddae193732486435b15ea5757b56a9e1a2096c06dd5f3143`.
- `ubs crates/ft-kernel-cpu/src/lib.rs` completed with no critical findings; it reported broad pre-existing warning inventory. Artifact SHA-256 `90fd487c83cdb12a14fec1ebffc59c525a35c950135192b2a3dc28e291316c79`.
