# ft-device Guard Profile Pass 23

- Bead: `frankentorch-08g3`
- Date: 2026-06-02
- Skill loop: `/profiling-software-performance` plus `/extreme-software-optimization`
- Crate: `ft-device`
- Outcome: no optimization lever retained; both tested levers regressed the profiled scenarios.
- Golden output: `artifacts/optimization/golden_outputs/ft_device_guard_pass23.txt`
- Golden sha256: `27675514b55b508afc66fdcc3e59c977031e742ae604ef7316a34508eef427f8`

## Scenario

Direct crate-local device precondition checks and fail-closed error rendering:

- `device_guard/ensure_tensor_device_match_65536`
- `device_guard/ensure_same_device_match_65536`
- `device_guard/mismatch_display_65536`

The guard success paths model dispatch precondition checks. The mismatch display
path models fail-closed conformance and diagnostic reporting.

## Ranked Hotspot Table

| Rank | Location | Metric | Value | Category | Evidence |
|------|----------|--------|-------|----------|----------|
| 1 | `DeviceError::fmt` mismatch rendering | Criterion p50 | `1.7308 ms` | CPU/formatting | `rch exec -- cargo bench -p ft-device --bench device_bench -- device_guard/mismatch_display_65536 --warm-up-time 1 --measurement-time 5 --sample-size 20` on `vmi1149989` |
| 2 | `ensure_same_device` match path | Criterion p50 | `44.236 us` | CPU/precondition | `rch exec -- cargo bench -p ft-device --bench device_bench -- device_guard --warm-up-time 1 --measurement-time 5 --sample-size 20` on `vmi1227854` |
| 3 | `DeviceGuard::ensure_tensor_device` match path | Criterion p50 | `42.008 us` | CPU/precondition | same rch Criterion run on `vmi1227854` |

## Hypothesis Ledger

| Hypothesis | Verdict | Evidence |
|------------|---------|----------|
| Cross-crate inline hints on tiny guard APIs reduce dispatch precondition overhead. | rejected | Success-path p50 regressed to `76.898 us` and `76.650 us` on `vmi1153651`; no scoreable win. |
| Manual `Device` label writes avoid Debug formatting overhead in mismatch display. | rejected | Mismatch-display p50 regressed from `1.7308 ms` on `vmi1149989` to `3.6230 ms` on `vmi1153651`; confidence capped for cross-worker comparison, but direction and magnitude were not keepable. |

## Baseline And Trial Runs

Pre-lever success-path baseline:

```text
device_guard/ensure_tensor_device_match_65536
time: [41.093 us 42.008 us 42.870 us]

device_guard/ensure_same_device_match_65536
time: [41.401 us 44.236 us 47.656 us]
```

Rejected inline trial:

```text
device_guard/ensure_tensor_device_match_65536
time: [74.864 us 76.898 us 79.062 us]

device_guard/ensure_same_device_match_65536
time: [73.840 us 76.650 us 78.917 us]
```

Pre-lever mismatch-display baseline:

```text
device_guard/mismatch_display_65536
time: [1.6516 ms 1.7308 ms 1.7976 ms]
```

Rejected manual-label trial:

```text
device_guard/mismatch_display_65536
time: [3.3689 ms 3.6230 ms 3.9610 ms]
```

## Isomorphism Proof

- Ordering preserved: yes; no retained hot-path behavior change.
- Tie-breaking unchanged: N/A.
- Floating-point: N/A.
- RNG seeds: N/A.
- Device mismatch semantics: exact `DeviceError::Display` string remains frozen by the golden fixture.
- Golden outputs: `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed.

## Decision

No optimization lever met the score threshold. The retained diff is the
crate-local benchmark and golden fixture so future work has a profile-backed
starting point and does not retry the rejected inline or manual-label levers.
