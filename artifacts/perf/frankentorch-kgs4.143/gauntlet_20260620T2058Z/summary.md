# frankentorch-kgs4.143 BatchNorm2d f32 Automatic tensor_sum Shortcut

## Objective

Target the f32 BatchNorm2d scalar-loss training gap by partial-evaluating the
ordinary API pattern:

`functional_batch_norm2d(...).tensor_sum().backward()`

When the BatchNorm output does not retain gradients and has no hooks, the final
source routes `tensor_sum` through the existing scalar-loss BatchNorm2d backward
instead of building the generic `Sum` autograd edge. Retained-grad and hook
cases fall back to the materialized output-gradient path.

## Result

Verdict: keep as an internal FrankenTorch speedup, record as a PyTorch loss.

| Row | Worker/host | Median | PyTorch median | Ratio vs PyTorch |
|---|---:|---:|---:|---:|
| Baseline ordinary materialized | rch `hz1` | `203.33 ms` | local `8.101561 ms` | `25.10x` slower |
| Baseline explicit scalar-sum | rch `hz1` | `55.138 ms` | local `8.101561 ms` | `6.81x` slower |
| Temporary disabled ordinary | rch `vmi1152480` | `166.77 ms` | local `7.705096 ms` | `21.64x` slower |
| Enabled automatic ordinary | rch `vmi1152480` | `117.96 ms` | local `7.705096 ms` | `15.31x` slower |
| Temporary disabled scalar-sum control | rch `vmi1152480` | `100.95 ms` | local `7.705096 ms` | `13.10x` slower |
| Enabled scalar-sum control | rch `vmi1152480` | `103.35 ms` | local `7.705096 ms` | `13.41x` slower |

Same-worker internal A/B on `vmi1152480`: ordinary path `166.77 ms` disabled to
`117.96 ms` enabled, runtime ratio `0.707x`, speedup `1.41x`. The explicit
scalar-sum control was stable (`100.95 ms` disabled vs `103.35 ms` enabled), so
the measured win is on the automatic ordinary API path.

Remote PyTorch arms failed on rch workers with `ModuleNotFoundError: No module
named 'torch'`; ratio-vs-PyTorch evidence uses the local CPU oracle in
`/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python`.

Win/loss/neutral vs PyTorch for this bead: `0W / 1L / 0N`.

## Gates

- `AGENT_NAME=IvoryDeer CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo test -p ft-api functional_batch_norm2d_f32_tensor_sum --lib --profile release -- --nocapture`: passed, 2/0.
- `AGENT_NAME=IvoryDeer CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo test -p ft-api functional_batch_norm2d_f32_sum --lib --profile release -- --nocapture`: passed, 2/0.
- `AGENT_NAME=IvoryDeer CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench --profile release`: passed.
- `AGENT_NAME=IvoryDeer CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings`: passed.
- `AGENT_NAME=IvoryDeer CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo test -p ft-conformance --profile release`: passed, full conformance crate and sub-suites green.
- `git diff --check`: passed.
- `AGENT_NAME=IvoryDeer CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo fmt --check -p ft-api`: reports broad pre-existing rustfmt diffs in ft-api benches/examples and old hunks in the giant `lib.rs`; no formatting rewrite was kept in this perf commit.

## Raw Evidence

- `baseline_rch_batch_norm2d_f32.log`
- `baseline_local_pytorch_batch_norm2d_f32_5x40.log`
- `after_rch_batch_norm2d_f32.log`
- `disabled_rch_batch_norm2d_f32_vmi1152480.log`
- `after_local_pytorch_batch_norm2d_f32_5x40.log`
- `test_ft_api_batch_norm2d_auto_shortcut.log`
- `test_ft_api_batch_norm2d_sum_existing.log`
- `check_ft_api_bench.log`
- `clippy_ft_api_bench.log`
- `test_ft_conformance_release.log`
- `fmt_check_ft_api.log`
