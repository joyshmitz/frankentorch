# Negative Evidence Ledger: frankentorch-kfdnn

Agent: IvoryDeer / cod-b
Run: gauntlet_20260620T044748Z
Head: 66bc9fd4bfe6a801724df402191084f55799791f
Target: `gauntlet_max_pool3d_grad/frankentorch_fused_sum_loss` versus `pytorch_2_12_cpu`

## Candidate Rejected

Lever: packed u16 argmax sidecar for the scalar-loss `max_pool3d_sum` autograd path.

Rationale tested: reduce saved-tensor sidecar traffic by packing four plane-local `u16` argmax offsets into one `f64` word, while preserving bit-exact first-tie behavior and the existing saved tensor contract.

Implementation status: reverted before commit. Correctness probes passed, but the target benchmark did not improve.

## Commands

Baseline:

```sh
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool3d --noplot \
> artifacts/perf/frankentorch-kfdnn/gauntlet_20260620T044748Z/baseline_local_max_pool3d_20260620T044748Z.log 2>&1
```

Candidate correctness:

```sh
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
rch exec -- cargo test -p ft-kernel-cpu max_pool3d_sum_packed_u16_backward_matches_materialized_bits -- --nocapture \
> artifacts/perf/frankentorch-kfdnn/gauntlet_20260620T044748Z/test_kernel_max_pool3d_packed_u16_20260620T044748Z.log 2>&1

CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
rch exec -- cargo test -p ft-api functional_max_pool3d_sum_matches_pool_sum_backward_bits -- --nocapture \
> artifacts/perf/frankentorch-kfdnn/gauntlet_20260620T044748Z/test_api_max_pool3d_packed_sum_bits_20260620T044748Z.log 2>&1
```

After:

```sh
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool3d --noplot \
> artifacts/perf/frankentorch-kfdnn/gauntlet_20260620T044748Z/after_local_max_pool3d_packed_u16_20260620T044748Z.log 2>&1
```

## Target Result

| Row | Baseline median | After median | Delta verdict |
| --- | ---: | ---: | --- |
| `frankentorch_fused_sum_loss` | 5.6985 ms | 5.7355 ms | neutral / no gain, p = 0.37 |
| `pytorch_2_12_cpu` | 1.8203 ms | 2.1262 ms | noisy, p = 0.31 |

Ratio versus PyTorch:

| Run | FrankenTorch median | PyTorch median | Ratio |
| --- | ---: | ---: | ---: |
| Baseline | 5.6985 ms | 1.8203 ms | 3.130x slower |
| After | 5.7355 ms | 2.1262 ms | 2.697x slower |

The apparent ratio shift is not accepted as a gain because PyTorch moved noisily while the FrankenTorch target row worsened by median and stayed statistically neutral.

Win/loss/neutral versus PyTorch for the target row: 0 / 1 / 0.

## Stage Evidence

| Stage | Baseline median | After median | Verdict |
| --- | ---: | ---: | --- |
| setup tensor | 211.29 us | 235.67 us | regression, but outside candidate path |
| forward only | 1.6122 ms | 1.6091 ms | neutral |
| sum only | 782.70 us | 803.45 us | neutral |
| backward only | 5.3919 ms | 5.4074 ms | neutral |
| raw kernel forward with indices | 778.05 us | 759.46 us | improved, not the packed fused path |
| raw kernel backward from indices | 1.5926 ms | 1.5164 ms | improved, not the packed fused path |

## Decision

Reject and revert.

Why: the user-facing target is still a PyTorch loss, and the fused row did not improve. This rules out compact saved sidecar representation as the next dominant lever for `frankentorch-kfdnn`.

Next route: `frankentorch-maxpool3d-scalar-loss-grad-buffers-7wru6` attacks scalar-loss autograd/session overhead rather than sidecar density. The measured residual is concentrated in the fused backward/session path, so the next bead should target lazy gradient slot creation, duplicate persistent gradient buffers, or a fused primitive that bypasses generic tape accumulation for `loss = sum(max_pool3d(...))` while preserving deterministic autograd evidence.
