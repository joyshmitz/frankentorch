# frankentorch-kgs4.108 closeout: lazy Gelu backward accumulation

## Decision

KEEP. The Gelu tensor backward rule now accumulates the exact same derivative directly into the destination gradient buffer with `accumulate_tensor_gradient_zip_map`, avoiding the temporary contribution `Vec<f64>`.

Score: `2.43 = Impact 1.280 * Confidence 0.95 / Effort 0.50`.

## Profile-backed target

- Parent route: `frankentorch-kgs4` no-gaps performance campaign.
- Hot row: `activation_backward/gelu_chain_16x65536`.
- Prior profile evidence: `artifacts/perf/frankentorch-ivorydeer-autograd-reprofile-20260615/baseline_backward_bench_vmi1227854.log` identified Gelu-chain backward as an autograd activation hotspot.
- Execution mode: local Cargo/Criterion only because the 2026-06-16 `ts1` override forbids waiting on remote `rch`.

## Baseline

Command:

```bash
env -u RCH_REQUIRE_REMOTE CARGO_TARGET_DIR=/data/tmp/frankentorch-a7xya-local-target cargo bench -j 1 -p ft-autograd --bench backward_bench -- gelu_chain_16x65536 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Criterion row:

```text
activation_backward/gelu_chain_16x65536
time: [48.739 ms 49.474 ms 49.975 ms]
```

Artifact: `pass1_local_baseline_gelu_backward.log`

SHA-256: `505502d987d0dcfd8618c564ffb9f6a66123b3f7c60d3afa9d42b277c32efa0e`

## Lever

One lever only:

- Replaced `tensor_backward_zip_map(...); accumulate_tensor_gradient(..., &contrib)` in `TensorNodeOp::Gelu` with direct `accumulate_tensor_gradient_zip_map(...)`.
- No changes to the public API, dispatch, benchmark, dtype conversion, dependency queue, or create-graph path.

## Isomorphism proof

Behavior invariants preserved:

- Ordering: same ascending zip over `incoming` and `input_values`.
- Floating point: same derivative formula, same constants, same `exp` and `libm::erf` calls per element.
- Tie-breaking/RNG: none present before or after.
- Gradient accumulation: new branched test proves existing gradient contents are combined with the Gelu derivative bit-for-bit.
- Evidence path: `TensorBackwardStep` rule and dependency completion order unchanged.

Focused proof:

```bash
env -u RCH_REQUIRE_REMOTE CARGO_TARGET_DIR=/data/tmp/frankentorch-a7xya-local-target cargo test -j 1 -p ft-autograd tensor_gelu --lib -- --nocapture
```

Result: 4/4 tests passed, including `tensor_gelu_branch_accumulates_existing_gradient_bit_exact`.

Artifact: `pass5_proof_tensor_gelu_final.log`

SHA-256: `568cbf44e4738678afa764b421666b2d5346519c7c90200acad2bbeb1af8546c`

## Rebench

Command:

```bash
env -u RCH_REQUIRE_REMOTE CARGO_TARGET_DIR=/data/tmp/frankentorch-a7xya-local-target cargo bench -j 1 -p ft-autograd --bench backward_bench -- gelu_chain_16x65536 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Criterion row:

```text
activation_backward/gelu_chain_16x65536
time: [37.500 ms 38.641 ms 40.729 ms]
change: [-23.664% -21.083% -17.490%], p = 0.00
```

Artifact: `pass3_local_rebench_gelu_backward.log`

SHA-256: `4bf4ce594e76b3b2be18a05d3e7491e58540f3be3fd12c15cf387acb4923dbe8`

## Quality gates

- `cargo check -j 1 -p ft-autograd --all-targets`: passed.
  - Artifact: `pass4_check_ft_autograd_all_targets.log`
  - SHA-256: `931aa0eeda1b7d6cfec44b003a556fd3e9e9e847962948eb7fb1ce200d8d0a06`
- `git diff --check`: passed.
  - Artifact: `pass5_git_diff_check.log`
  - SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `cargo fmt -p ft-autograd -- --check`: blocked by pre-existing unrelated formatting drift at `crates/ft-autograd/src/lib.rs:10280`, `21443`, and `21554`.
  - Artifact: `pass6_fmt_ft_autograd_check_pipefail.log`
  - SHA-256: `b8519846f3ea47db12f46f02b5f1d3ce470e99ae3b2523c05cc1032402d9d2f5`
- `cargo clippy -j 1 -p ft-autograd --lib --no-deps -- -D warnings`: blocked by pre-existing unrelated lints at `crates/ft-autograd/src/lib.rs:10034` and `18839`.
  - Artifact: `pass6_clippy_ft_autograd_lib.log`
  - SHA-256: `ee51661f2e04b4a301745cf9424821cc9f7b75336ef02bef7028db32317ad79b`

The residual fmt/clippy blockers are outside this Gelu lever and were not repaired in this commit to preserve the one-lever performance discipline.
