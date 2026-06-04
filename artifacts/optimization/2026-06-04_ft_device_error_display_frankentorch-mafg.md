# ft-device DeviceError display names - frankentorch-mafg

## Target

- Bead: `frankentorch-mafg`
- Crate: `ft-device`
- Benchmark: `device_guard/mismatch_display_65536`
- Lever: render `DeviceError::Mismatch` device names through a fixed local match instead of `Debug` formatting.

## Profile-backed baseline

`br ready --json` returned no ready perf beads, so this was a fallback profile pass on an unowned crate surface. Clean `HEAD` baseline via `rch` on worker `ts2`:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-device --bench device_bench -- device_guard/mismatch_display_65536 --warm-up-time 1 --measurement-time 5 --sample-size 20
device_guard/mismatch_display_65536 time: [2.6904 ms 2.7043 ms 2.7147 ms]
```

The profile showed the match-path device guards at about 27-28 us for 65,536 iterations, while mismatch display formatting was over 2.7 ms on the same worker. The hot path was repeated `Debug` formatting for the two `Device` enum values inside the error string.

## Result

Same benchmark, same worker, current candidate:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-device --bench device_bench -- device_guard/mismatch_display_65536 --warm-up-time 1 --measurement-time 5 --sample-size 20
device_guard/mismatch_display_65536 time: [1.9664 ms 1.9800 ms 1.9961 ms]
```

P50 speedup:

```text
2.7043 ms / 1.9800 ms = 1.366x
```

Score: `Impact 2 * Confidence 3 / Effort 1 = 6.0`; keep.

## Isomorphism proof

- Display string unchanged: `device_error_display` now asserts exact output, `device mismatch: expected Cpu, got Cuda`.
- Error variant values unchanged: `DeviceError::Mismatch { expected, actual }` remains the only variant and carries the same enum values.
- Ordering/tie-breaking unchanged: N/A. This formats one error value.
- Floating-point unchanged: N/A.
- RNG unchanged: yes. No random state is read or modified.
- Golden output: `artifacts/optimization/golden_outputs/ft_device_error_display_frankentorch-mafg.txt`

## Validation

Passed:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-device device_error_display -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-device --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-device --all-targets -- -D warnings
cargo fmt -p ft-device --check
```
