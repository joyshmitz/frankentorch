# ft-autograd no-hook backward clone pass rejected

Bead: `frankentorch-kgs4.32`

## Target

Profile-backed fallback target after `br ready --json` stayed empty and active perf beads were already claimed on `ft-kernel-cpu` and `ft-optim`.

- SLO evidence: `artifacts/phase2c/performance/perf_slo_measurement_v2.json` reports representative backward p95 ratio `2.138` vs budget `1.350`.
- Allocation evidence: the same report notes allocation counts cover `tensor_backward`.
- Criterion baseline command: `rch exec -- cargo bench -p ft-api --bench ops_bench -- backward_matmul/size/128 --warm-up-time 1 --measurement-time 5 --sample-size 20`.
- Baseline worker/result: `vmi1227854`, `[1.2332 ms 1.2663 ms 1.2972 ms]`.

## Lever Attempted

Branch `TensorTape::backward_with_options` so nodes without tensor hooks avoid the `apply_tensor_hooks` no-hook `incoming.to_vec()` plus caller writeback clone. Hooked nodes kept the existing hook application, length validation, ordering, and writeback behavior.

## Behavior Proof During Attempt

- Focused golden test passed: `rch exec -- cargo test -p ft-autograd tensor_backward_no_hook_clone_fast_path_golden_output_is_stable -- --nocapture`.
- Golden fixture sha256: `a0520b910829e8f7d9b3f82e0d90fc87b1e0ab2b68e9566dbbbbc78ebb2670dd`.
- Ordering unchanged: dependency queue and execution order were not modified.
- Hook ordering unchanged: the golden covered two hooks in registration order.
- Floating-point unchanged: the lever touched no arithmetic.
- RNG unchanged: no random state or seeds are involved.

## Re-benchmark

- After command: `rch exec -- cargo bench -p ft-api --bench ops_bench -- backward_matmul/size/128 --warm-up-time 1 --measurement-time 5 --sample-size 20`.
- After worker/result: `vmi1153651`, `[2.7604 ms 2.8635 ms 2.9722 ms]`.

The after run did not prove a win. Even allowing for cross-worker noise, the measured p50 regressed from `1.2663 ms` to `2.8635 ms`.

## Verdict

Rejected and manually reverted. No `ft-autograd` source change is kept.

Score: impact `-2` x confidence `2` / effort `1` = `-4.0`.
