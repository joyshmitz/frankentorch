# frankentorch-gj8t mixed-precision iterative-refinement solve

## Target

- Bead: `frankentorch-gj8t`
- Primitive: safe-Rust mixed-precision iterative refinement for no-grad f64 `linalg.solve`
- Profile-backed hotspot: `ft-kernel-cpu` LU factor/solve on 512x512 with 32 RHS columns
- Worker: `ts1`

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- lu_solve_f64_512x512_rhs32 --warm-up-time 1 --measurement-time 5 --sample-size 10
```

Result:

- `lu_solve_f64_512x512_rhs32`: `[77.413 ms 92.863 ms 109.42 ms]`

## Change

One lever:

- Add an f32 LU factorization converted from f64 input.
- Solve initial f64 RHS values against the f32 LU factor.
- Apply two f64 residual-correction solves.
- Route only no-grad `ft-api::tensor_linalg_solve` through the mixed-refinement helper.

Strict fallback:

- `n < 128` uses the previous f64 factor/solve path.
- Singular or near-singular f32 factors fall back to checked f64.
- If final residual exceeds `max(abs(B), 1.0) * 1e-9`, fall back to checked f64.
- Requires-grad solve path is unchanged and still composes through inverse/matmul for autograd.

## After

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- lu_solve_mixed_refine_512x512_rhs32 --warm-up-time 1 --measurement-time 5 --sample-size 10
```

Result:

- `lu_solve_mixed_refine_512x512_rhs32`: `[28.368 ms 28.561 ms 28.980 ms]`

Speedup:

- Same-worker median: `92.863 ms -> 28.561 ms`
- Ratio: `3.25x`

Score:

- `Impact 3.25 x Confidence 0.90 / Effort 1.4 = 2.09`
- Verdict: keep

## Isomorphism proof

- Ordering/tie-breaking: f64 fallback keeps existing LU pivot scan semantics for small, singular, near-singular, and residual-failing cases. The mixed path is deterministic and uses a fixed first-greater pivot scan in f32; no maps/sets or unstable output ordering are introduced.
- Floating point: the no-grad large-system path is not bit-exact by design; contract is solve equivalence under residual/reference tolerance. The proof run measured `max_diff=4.7704895589362195e-18` versus the existing f64 LU solve and `max_residual=4.440892098500626e-16` on the deterministic 128x128 RHS4 proof matrix.
- RNG: no RNG state is read or modified in the implementation or benchmarks; benchmark/proof matrices are deterministic formulas.
- Autograd: requires-grad `tensor_linalg_solve` route is unchanged, preserving the existing gradient construction and DAC path.
- Error behavior: `solve_singular_errors_loud` initially caught a parity gap; fallback now checks f64 LU diagonal singularity before solve, and the full `solve_` ft-api test filter passes.

## Golden/output proof

Evidence files:

- `baseline_lu_solve_f64_512_rhs32.txt`
- `after_lu_solve_mixed_refine_512_rhs32.txt`
- `proof_scratch_mixed_refine_residual.txt`
- `test_kernel_mixed_refine_rerun.txt`
- `test_ft_api_solve_filter_rerun.txt`
- `check_ft_kernel_cpu.txt`
- `check_ft_api.txt`
- `clippy_ft_kernel_cpu_rerun.txt`

Golden proof values:

- `mixed_refine_proof max_diff=4.7704895589362195e-18`
- `mixed_refine_proof max_residual=4.440892098500626e-16`

## Gates

Passed:

- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo check -p ft-kernel-cpu`
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo check -p ft-api`
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-kernel-cpu lu_solve_mixed_refine_matches_f64_reference_and_residual -- --nocapture`
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-api solve_ -- --nocapture`
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`
- `git diff --check`

Blocked by pre-existing unrelated repo debt:

- `cargo fmt -p ft-kernel-cpu -p ft-api -- --check` reports broad existing formatting drift in ft-api benches/examples and unrelated ft-kernel-cpu lines. Touched lines from this lever were manually aligned to rustfmt output.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-api --all-targets -- -D warnings` reports 200+ existing ft-api warnings across the 100k-line file.
- `ubs crates/ft-api/src/lib.rs crates/ft-kernel-cpu/src/lib.rs crates/ft-kernel-cpu/benches/linalg_bench.rs` exited nonzero after 312s from broad existing inventories; its internal fmt/clippy/build subchecks were clean.

## Next primitive

Re-profile ready perf beads after closing `frankentorch-gj8t`. Avoid the peer-owned eigensolver bead and target the next independent BLAS/LAPACK-class safe-Rust primitive.
