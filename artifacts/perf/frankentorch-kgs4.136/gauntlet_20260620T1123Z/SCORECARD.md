# frankentorch-kgs4.136 Scorecard

## Performance

- Direct rch diagnostic: `10.80 ms` existing fused -> `1.66 ms` scalar-sum (`6.50x` faster).
- rch Criterion target shape: `114.23 ms` existing fused -> `78.166 ms` scalar-sum (`1.46x` faster).
- PyTorch ratio: local CPU PyTorch `5.605736 ms/iter`; scalar-sum remains `13.94x` slower.

## Correctness

- `cargo test -p ft-kernel-cpu batch_norm_f32_scalar_backward_matches -- --nocapture`: passed, 2 tests.
- `cargo test -p ft-api functional_batch_norm2d_f32_sum --lib`: passed.
- `cargo test -p ft-conformance strict_scheduler`: passed.

## Build / Lint

- `cargo check -p ft-api --all-targets`: passed, existing unrelated `hessian_probe.rs` warning.
- `cargo check -p ft-kernel-cpu --all-targets`: passed, existing unrelated `gemm_golden.rs` warnings.
- `cargo clippy -p ft-api --lib -- -D warnings`: passed.
- `cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings`: passed.
- `cargo clippy -p ft-api --example batch_norm_f32_grad_ab -- -D warnings`: passed.
- `cargo clippy -p ft-kernel-cpu --lib -- -D warnings`: passed.
- Targeted small-file `rustfmt --check`: passed after formatting `batch_norm_f32_grad_ab.rs`.
- `git diff --check`: passed.
- `ubs <scoped files>`: timed out after 240 seconds with no findings emitted beyond `Scanning rust...`.
- Broad `cargo fmt --check -p ft-api -p ft-kernel-cpu`: blocked by pre-existing unrelated rustfmt drift.
