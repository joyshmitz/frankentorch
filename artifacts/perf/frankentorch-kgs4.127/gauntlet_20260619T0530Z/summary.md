# frankentorch-kgs4.127 SmoothL1 one-sided grad BOLD-VERIFY summary

Agent: `IvoryDeer`
Target: `frankentorch-kgs4.127`
Workload: `smooth_l1/grad_8m`, 8,388,608 f64 elements, mean reduction,
forward plus backward.

## Lever

Reduced f64 SmoothL1 already avoided the materialized per-element loss tensor.
This pass removes the next obvious allocation: when only the input requires a
gradient, do not allocate or fill the unused target-gradient vector. The target
only case gets the symmetric helper.

## Timings

| Row | Host/worker | Median or Criterion center | Role |
|---|---:|---:|---|
| PyTorch `2.12.1+cpu` | local `thinkstation1` | `360.7852805 ms` | oracle |
| FrankenTorch pre-lever | local `thinkstation1` | `746.26 ms` | decisive baseline |
| FrankenTorch candidate | local `thinkstation1` | `647.44 ms` | decisive candidate |
| FrankenTorch pre-lever | rch `ovh-a` | `674.81 ms` | remote baseline, different-worker only |
| FrankenTorch candidate | rch `hz1` | `774.85 ms` | remote build/bench, different-worker only |
| FrankenTorch candidate | rch `vmi1152480` | `619.16 ms` | remote build/bench, different-worker only |

Same-host local A/B speedup: `1.15x`.
Candidate ratio vs local PyTorch: `1.79x` slower.
Win/loss/neutral vs PyTorch: `0W / 1L / 0N`.

`RCH_WORKER=ovh-a` was requested for the second candidate remote run, but this
rch version selected `vmi1152480`; remote rows are therefore build and routing
evidence, not same-worker proof.

## Correctness and Gates

- `rch exec -- cargo test -p ft-kernel-cpu smooth_l1_backward_reduced_one_sided_helpers_match_full_bits -- --nocapture`: passed.
- `rch exec -- cargo test -p ft-api smooth_l1_loss_reduced_grad_skips_unneeded_one_sided_gradient -- --nocapture`: passed.
- `rch exec -- cargo check -p ft-api`: passed.
- `rch exec -- cargo clippy -p ft-api -- -D warnings`: passed.
- `rch exec -- cargo clippy -p ft-kernel-cpu -- -D warnings`: passed.
- `git diff --check`: passed.
- `rustfmt --edition 2024 --check crates/ft-api/src/lib.rs crates/ft-kernel-cpu/src/lib.rs`: failed on broad pre-existing formatting drift in these large files.
- `ubs crates/ft-api/src/lib.rs crates/ft-kernel-cpu/src/lib.rs`: completed after 545s with the known broad inventory in these files; no SmoothL1-specific blocker was identified.

Hardware profiling was blocked by `/proc/sys/kernel/perf_event_paranoid=4`.

## Verdict

Keep. This is a measured internal win and it narrows the SmoothL1 gap, but
PyTorch still wins decisively. Do not retry another one-sided reduced-gradient
wrapper. The next SmoothL1 bead should target allocator lifetime, tape edge
collapse, input/RNG setup, SIMD or branchless gradient generation, or a fused
train-step path with fresh head-to-head evidence.
