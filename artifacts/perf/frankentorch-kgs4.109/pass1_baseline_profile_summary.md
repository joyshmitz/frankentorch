# frankentorch-kgs4.109 Pass 1 Baseline/Profile Confirmation

Timestamp: 2026-06-17T02:34:09-04:00
Agent: IvoryDeer
Scope: measurement only; no Rust source edits.

## Bead

`frankentorch-kgs4.109`: `[perf][no-gaps] ft-api/ft-kernel-cpu: direct f64 max_pool1d grad route`

The bead is profile-backed by `artifacts/perf/frankentorch-ftapi-train-reprofile-20260616/baseline_train_hotspots.log`, where `conv1d_family/max_pool1d_grad` reported:

```text
time: [293.87 ms 306.70 ms 319.85 ms]
```

That older routing run used a short 10-sample rch-era broad profile command. This pass captured a current local, crate-scoped Criterion baseline because ts1/rch is offline for this campaign.

## Command

```bash
CARGO_TARGET_DIR=/tmp/frankentorch-kgs4.109-target cargo bench -p ft-api --bench ops_bench -- max_pool1d_grad 2>&1 | tee artifacts/perf/frankentorch-kgs4.109/pass1_local_baseline_max_pool1d_grad_20260617T0633Z.log
```

## Current Baseline

Benchmark row: `conv1d_family/max_pool1d_grad`

Criterion output:

```text
time: [447.62 ms 451.67 ms 456.11 ms]
Found 7 outliers among 100 measurements (7.00%)
  1 (1.00%) low mild
  4 (4.00%) high mild
  2 (2.00%) high severe
```

Structured estimates:

- mean point estimate: 451.67176488 ms
- mean 95% CI: [447.6180118 ms, 456.11244516 ms]
- median point estimate: 449.6323125 ms
- median 95% CI: [446.987332 ms, 453.322624 ms]

## Context Inspected

- `crates/ft-api/benches/ops_bench.rs`: `max_pool1d_grad` uses `[N,C,L]=[8,64,8192]`, `kernel_size=2`, `stride=2`, random f64 input requiring grad, sum loss, then backward.
- `crates/ft-api/src/lib.rs`: `functional_max_pool1d` validates 1D pooling shape, preserves unbatched `[C,L]` by unsqueeze/pool/squeeze, reshapes `[N,C,L]` to `[N,C,1,L]`, calls `functional_max_pool2d`, then reshapes back to `[N,C,out_l]`.
- `crates/ft-kernel-cpu/src/lib.rs`: f64 max-pool2d grad path uses ascending window scans with `if v > m`, preserving first-argmax tie behavior, saves plane-local arg offsets, and scatters gradients in output order.

## Artifacts

- Log: `artifacts/perf/frankentorch-kgs4.109/pass1_local_baseline_max_pool1d_grad_20260617T0633Z.log`
  - SHA-256: `f585c2b66f0b5ba5056d481d726e2e1d368b093cc318165960ecee6a9d25f049`
- Estimates JSON: `artifacts/perf/frankentorch-kgs4.109/pass1_local_baseline_max_pool1d_grad_20260617T0633Z.estimates.json`
  - SHA-256: `284e0488294366e51e49f9d402ff0cd1f56166216cf9b8fb09af61aa78bef9ff`
- Sample JSON: `artifacts/perf/frankentorch-kgs4.109/pass1_local_baseline_max_pool1d_grad_20260617T0633Z.sample.json`
  - SHA-256: `b8ca0d5de090f244262c47f4e08a55340d82dd102f1fe842b70d186601f708eb`

## Candidate Recommendation

Proceed to a single pass-2 lever: add a direct f64 `max_pool1d` route for 3-D `[N,C,L]` inputs instead of reshaping through height-1 `max_pool2d`. The later implementation should keep the existing fallback behavior for non-f64 and unsupported cases.

The direct route should preserve:

- first-argmax tie policy: update arg only on `v > m`
- ascending window scan order: output position outer loop, kernel offset inner loop
- gradient scatter order for overlapping windows: output positions in ascending order per `(N,C)` plane
- dtype and error behavior
- unbatched `[C,L]` wrapper semantics
- RNG absence

Initial score outlook: likely worth pass 2. The baseline is large and the candidate is a narrow analogue of the recently kept avg-pool1d route, but the keep/reject decision must wait for same-local-target rebench evidence after exactly one implementation lever.
