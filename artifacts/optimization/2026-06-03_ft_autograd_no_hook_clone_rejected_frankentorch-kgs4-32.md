# ft-autograd No-Hook Gradient Clone Rejected Pass

- Bead: `frankentorch-kgs4.32`
- Target: `TensorTape::backward_with_options` hookless gradient handling.
- Lever tested: skip `apply_tensor_hooks` for nodes without tensor hooks and use the already-stored gradient clone.
- Status: rejected; no source change retained.

## Profile Target

Profile evidence came from the backward SLO miss in
`artifacts/phase2c/performance/perf_slo_measurement_v2.json` and the existing
`backward_matmul/size/128` Criterion target.

Baseline control command:

```text
rch exec -- cargo bench -p ft-api --bench ops_bench -- backward_matmul/size/128 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Baseline control result on `vmi1293453`:

```text
backward_matmul/size/128 time: [1.1812 ms 1.2021 ms 1.2250 ms]
```

Candidate after result on `vmi1149989`:

```text
backward_matmul/size/128 time: [1.0649 ms 1.0879 ms 1.1109 ms]
```

Candidate repeat result on `vmi1149989`:

```text
backward_matmul/size/128 time: [1.1402 ms 1.3049 ms 1.4528 ms]
```

The first after run was a small cross-worker p50 improvement, but the repeat
after run on the same worker regressed relative to the baseline p50. The result
does not prove a real win.

## Isomorphism Notes

The discarded candidate did not change execution queue order, dependency
bookkeeping, floating-point arithmetic, RNG, or tie-breaking. Hooked nodes kept
the existing `apply_tensor_hooks` ordering and shape-validation path. Hookless
nodes would have skipped only the no-hook `incoming.to_vec()` clone inside
`apply_tensor_hooks`.

A local golden fixture was generated for the discarded draft:

```text
a0520b910829e8f7d9b3f82e0d90fc87b1e0ab2b68e9566dbbbbc78ebb2670dd  artifacts/optimization/golden_outputs/ft_autograd_no_hook_backward_clone_frankentorch-kgs4-32.txt
```

No source diff was retained because the benchmark failed the score gate.

## Score

- Impact: 0
- Confidence: 1
- Effort: 1
- Score: 0.0

Rejected below the required 2.0 threshold.
