# cod-b cdist p=2 f32 panel-packing experiment

Date: 2026-06-23
Agent: QuietMeadow
Bead/thread: frankentorch-kgs4

Target residual: no-grad f32 `cdist(x1, x2, p=2)`, shape `2000x2000x100`.

Commands were crate-scoped only and used `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b`.

## Baseline

`cargo bench -p ft-api --bench cdist_bench -- cdist_p2_f32_fused/2000x2000x100 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot`

RCH worker: `vmi1149989`

FT baseline interval: `[5.3975 ms 6.9195 ms 8.9179 ms]`.

## Candidate

Temporary lever: pack each f32 `sgemm_bt_2d_parallel` transposed-RHS N panel into contiguous `[k,bj]` storage before tile calls, mirroring the normal-RHS 2-D GEMM panel layout.

Correctness gate:
`cargo test -p ft-kernel-cpu gemm_2d_parallel_is_bit_exact_vs_serial -- --nocapture`

Result: pass, 1 test.

After interval: `[5.7609 ms 5.9075 ms 6.1350 ms]`.

Criterion verdict: `No change in performance detected`, change `[-25.483% -5.6081% +18.849%]`, `p = 0.69`.

Decision: reverted as no-gain.

## PyTorch Comparator

Local oracle: `.venv-oracle`, torch `2.12.0+cpu`, 32 threads.

Same deterministic input generation as `cdist_bench.rs`.

`torch.cdist` result: min `1.3921 ms`, median `1.4731 ms`, p95 `1.5740 ms`, checksum `35011956.0`.

FT baseline ratio by midpoint vs PyTorch min: `6.9195 / 1.3921 = 4.97x SLOWER`.

Candidate ratio by midpoint vs PyTorch min: `5.9075 / 1.3921 = 4.24x SLOWER`, but the candidate was not kept because Criterion reported no reliable improvement.
