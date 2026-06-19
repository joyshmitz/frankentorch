# frankentorch-kgs4.132 Scorecard

Agent: IvoryDeer
Date: 2026-06-19
Worktree: `/data/projects/.scratch/frankentorch-cod-b-bold-verify-20260619T130454Z`
Baseline commit: `b5699cf7`
Target: `gauntlet_max_pool3d_grad`, f64 `[N,C,D,H,W]=[2,32,16,32,32]`, kernel `2x2x2`, stride `2x2x2`, forward max_pool3d, scalar sum, backward.
Lever: f64 custom autograd route that borrows input slices for forward setup and keeps the backward closure context-only/owned. This removes one owned input materialization on the max_pool3d forward path without reintroducing the prior borrowed-input backward reread.

## Headline

| Row | Baseline median | Candidate median | Verdict |
| --- | ---: | ---: | --- |
| Same-worker rch `hz2` FrankenTorch full row | 8.3166 ms | 5.4809 ms | WIN, 1.52x faster, -34.1%, Criterion p=0.00 |
| Same-worker rch `hz2` forward-only stage | 4.2347 ms | 1.5978 ms | WIN, 2.65x faster, -62.3%, Criterion p=0.00 |
| Local PyTorch ratio row, FrankenTorch | 6.4105 ms | 5.4457 ms | WIN vs local baseline, 1.18x faster by median |
| Local PyTorch ratio row, PyTorch | 1.8484 ms | 1.6027 ms | Reference remained faster |
| Ratio vs PyTorch | 3.47x slower | 3.40x slower | LOSS remains |

## Stage Breakdown, Same-Worker rch `hz2`

| Stage | Baseline median | Candidate median | Classification |
| --- | ---: | ---: | --- |
| setup tensor | 236.02 us | 234.60 us | neutral |
| forward only | 4.2347 ms | 1.5978 ms | win |
| sum only | 1.0322 ms | 1.0359 ms | neutral |
| backward only | 15.104 ms | 16.897 ms | neutral, p=0.31 |
| raw kernel forward with indices | 422.70 us | 453.65 us | neutral, p=0.15 |
| raw kernel backward from indices | 999.03 us | 1.2335 ms | neutral/noisy, p=0.05 not below threshold |

Stage W/L/N: 1 / 0 / 5.

## PyTorch Head-to-Head

Remote rch workers built and ran the FrankenTorch benchmark, but the PyTorch arm failed on `hz2` with `ModuleNotFoundError: No module named 'torch'`. The ratio-vs-PyTorch row therefore comes from the local PyTorch venv:

- Python: 3.13.7
- PyTorch: 2.12.1+cpu
- Threads: compute 32, interop 32
- Host: `thinkstation1`

Win/loss/neutral vs PyTorch: 0 / 1 / 0. The lever is a measured internal FrankenTorch win, not PyTorch dominance.

## Validation

- `rch exec -- cargo check -p ft-autograd -p ft-api --all-targets`: passed on `hz2`; existing warning in `crates/ft-api/examples/hessian_probe.rs`.
- `rch exec -- cargo test -p ft-autograd custom_function_borrowed_forward_owned_backward_uses_saved_context --lib`: passed on `hz2`.
- `rch exec -- cargo test -p ft-api functional_max_pool3d_grad_matches_finite_diff --lib`: passed on `hz1`.
- `rch exec -- cargo test -p ft-conformance strict_scheduler_conformance_is_green --lib`: passed on `ovh-a`.
- `rch exec -- cargo clippy -p ft-autograd --lib -- -D warnings`: passed on `hz2`.
- `rch exec -- cargo clippy -p ft-api --lib -- -D warnings`: passed on `hz2`.
- `git diff --check`: passed.

Known gate caveats:

- `cargo fmt --check` prints large pre-existing formatting diffs in unrelated crates. I did not run `cargo fmt` because that would modify peer-owned surfaces.
- `cargo clippy -p ft-autograd -p ft-api --all-targets -- -D warnings` fails on pre-existing example/test lint debt outside this lever, recorded in `clippy_ft_autograd_ft_api_all_targets.log`.
- `ubs` on the planned staged paths was interrupted after more than five minutes with no scanner result after initial Rust scan startup; partial output is recorded in `ubs_changed_files.log`. The pre-commit UBS hook also timed out on the large staged `ft-api/src/lib.rs` scan and suggested committing with `UBS_SKIP=1`.

## Evidence Logs

- `baseline_rch_max_pool3d.log`
- `candidate_rch_max_pool3d.log`
- `baseline_rch_max_pool3d_stage.log`
- `candidate_rch_max_pool3d_stage.log`
- `baseline_local_pytorch_max_pool3d.log`
- `candidate_local_pytorch_max_pool3d.log`
- `clippy_ft_autograd_lib.log`
- `clippy_ft_api_lib.log`
- `cargo_fmt_check.log`
- `clippy_ft_autograd_ft_api_all_targets.log`
- `ubs_changed_files.log`
