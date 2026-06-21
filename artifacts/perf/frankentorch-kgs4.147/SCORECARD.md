# frankentorch-kgs4.147 avg_pool2d scalar-loss scorecard

Date: 2026-06-21
Agent: cod-a

## Lever

Specialize f64 no-pad/no-ceil `sum(functional_avg_pool2d(...))` for the
gauntlet avg_pool2d workload. The measured internal implementation used the existing
optimized `avg_pool2d_forward_f64` plus `sum_tensor_contiguous_f64` forward
path and avoided the dense output-gradient buffer in backward with
`avg_pool2d_backward_scalar_f64`. This remote closeout is evidence-only; the helper is not present in the pushed source tree.

The allocation-free logical pooled-output range reducer was tried and removed:
it regressed the same-worker bench.

## Correctness

Command:

```text
env AGENT_NAME=cod-a CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo test -p ft-api --lib functional_avg_pool2d_sum_matches_pool_sum_backward_bits --release -- --nocapture
```

Final result:

```text
worker: vmi1153651
test tests::functional_avg_pool2d_sum_matches_pool_sum_backward_bits ... ok
test result: ok. 1 passed; 0 failed; 2339 filtered out
```

Conformance:

```text
env AGENT_NAME=cod-a CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo test -p ft-conformance --profile release

worker: vmi1153651
ft-conformance lib: 199 passed, 0 failed
binaries, integration tests, smoke tests, doctests: passed
```

## Bench Evidence

Rejected forward-deforestation candidate:

```text
worker: vmi1152480
gauntlet_avg_pool2d_grad/frankentorch_kgs4_112
  time: [37.718 ms 43.653 ms 46.733 ms]
gauntlet_avg_pool2d_grad/frankentorch_kgs4_147_fused_sum_loss
  time: [44.467 ms 78.599 ms 113.47 ms]
verdict: rejected, 78.599 / 43.653 = 1.80x slower
```

Final scalar-backward-only path:

```text
env AGENT_NAME=cod-a CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a RCH_WORKER=hz2 rch exec -- cargo bench --profile release -p ft-api --bench pytorch_gauntlet_bench -- gauntlet_avg_pool2d_grad/frankentorch --noplot

worker: hz2
gauntlet_avg_pool2d_grad/frankentorch_kgs4_112
  time: [8.6116 ms 10.685 ms 13.262 ms]
gauntlet_avg_pool2d_grad/frankentorch_kgs4_147_fused_sum_loss
  time: [4.7637 ms 6.2040 ms 8.8831 ms]
verdict: keep internally, 10.685 / 6.2040 = 1.72x faster
```

PyTorch sidecar:

```text
env FT_GAUNTLET_ITERS=40 FT_TORCH_THREADS=32 FT_TORCH_INTEROP_THREADS=32 /data/projects/.venvs/frankentorch-pytorch-cpu/bin/python crates/ft-api/benches/pytorch_avg_pool2d_grad.py

checksum=10.000000000000
0.107251892914 seconds / 40 = 2.681297323 ms/iter
```

Ratios:

- Final fused FT vs PyTorch: `6.2040 / 2.681297323 = 2.31x` slower.
- Same-worker materialized FT vs PyTorch: `10.685 / 2.681297323 = 3.98x` slower.
- PyTorch win/loss/neutral: `0W / 1L / 0N`.

## Verdict

Evidence-only closeout. The scalar-backward API and bench row were measured as an internal FrankenTorch win, but are not present in the pushed source tree. Do not count this as PyTorch dominance. Do not retry logical range-sum forward
deforestation without a same-worker microprofile showing it beats the existing
optimized materialized forward plus pairwise sum.
