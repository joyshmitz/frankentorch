# frankentorch-kgs4.135 Scorecard

Date: 2026-06-20

## Summary

`functional_group_norm_sum` is a measured internal keep for the f32 GroupNorm
scalar-loss lane, but it still loses to PyTorch.

| Metric | Result |
|---|---:|
| Direct rch scalar-sum vs prior fused path | `2.10 ms` vs `8.30 ms`, `3.96x` faster |
| Criterion scalar-sum vs materialized path | `8.9874 ms` vs `17.139 ms`, `1.91x` faster |
| Direct scalar-sum vs PyTorch fair best | `2.10 ms` vs `0.376163 ms`, `5.58x` slower |
| PyTorch head-to-head classification | `0W / 1L / 0N` |

## Commands

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a \
  rch exec -- cargo run --release -p ft-api --example group_norm_f32_grad_ab

CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a \
  rch exec -- cargo bench -p ft-api --bench ops_bench -- group_norm/grad_f32 \
  --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot

CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a \
  rch exec -- cargo test -p ft-api functional_group_norm_f32_sum --lib

CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a \
  rch exec -- cargo test -p ft-kernel-cpu \
  group_norm_f32_unit_dy_matches_general_reference_bits --lib

CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a \
  rch exec -- cargo test -p ft-conformance strict_scheduler
```

## Gates

- `ft-api` focused scalar-sum tests: passed, 2 tests.
- `ft-kernel-cpu` GroupNorm unit-dy guard: passed, 1 test.
- `ft-conformance` strict scheduler: passed.
- `cargo check -p ft-api --all-targets`: passed with an existing unrelated
  `hessian_probe.rs` warning.
- `cargo check -p ft-kernel-cpu --all-targets`: passed with existing unrelated
  `gemm_golden.rs` warnings.
- `cargo clippy -p ft-api --lib -- -D warnings`: passed.
- `cargo clippy -p ft-kernel-cpu --lib -- -D warnings`: passed.
- `git diff --check` on the scoped commit surface: passed.
- `ubs` on the scoped commit surface: interrupted after more than 3 minutes
  with no findings emitted, matching known large-file scanner timeout behavior.
- `cargo clippy --all-targets` remains blocked by existing unrelated test and
  example lint debt.
- `cargo fmt --check -p ft-api -p ft-kernel-cpu` remains blocked by existing
  unrelated rustfmt drift in large files.

## Release Action

Keep the scalar-sum API/kernel path. Route the remaining PyTorch gap to deeper
end-to-end mechanisms instead of another narrow GroupNorm branch:

- automatic scalar-loss specialization
- arena/bump allocation for tape and tensor buffers
- persistent f32 tensor storage
- dtype-boundary and conversion removal
- cache-blocked affine-gradient reductions
- scheduler and layout work with direct PyTorch-ratio proof
