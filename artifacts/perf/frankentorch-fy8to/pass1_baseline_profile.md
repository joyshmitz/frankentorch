# frankentorch-fy8to Pass 1 Baseline/Profile

Date: 2026-06-12

Scope: baseline/profile only for the strict-fallback Schur-window shift-list
kernel lane. No production source files were edited.

## RCH Status

This pass now has two remote evidence sets:

- Earlier same-worker command-run evidence on `vmi1149989`. The first command
  requested `vmi1227854`, but RCH selected `vmi1149989`; the paired rows were
  then kept on `vmi1149989`.
- Fresh historical-anchor rerun evidence requested `vmi1227854` with
  `RCH_REQUIRE_REMOTE=1`; every fresh rerun below selected remote worker
  `vmi1227854` and did not fall back locally.

The fresh `vmi1227854` rows are the best anchor for Pass 2 because this is the
worker used by most predecessor geev/eigvals evidence.

## Commands

Fresh `vmi1227854` rerun commands:

```bash
env RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- eigvals_f64_256x256
env RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- eig_f64_256x256
env RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo run --release -q -j 1 -p ft-kernel-cpu --example eigvals_golden
awk '/^frankentorch-l9xod eigvals_golden/{print; next} /^(eigvals_digest|eig_digest)=/{print}' artifacts/perf/frankentorch-fy8to/pass1_eigvals_golden_requested_vmi1227854.log > artifacts/perf/frankentorch-fy8to/pass1_eigvals_golden.strict.stdout
sha256sum artifacts/perf/frankentorch-fy8to/pass1_eigvals_golden.strict.stdout > artifacts/perf/frankentorch-fy8to/pass1_eigvals_golden.strict.stdout.sha256
env RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo run --release -q -j 1 -p ft-kernel-cpu --example eig_timing_probe
```

The `awk` and `sha256sum` commands were local artifact extraction/checksum
steps. The `eigvals_golden` program output was routed through the RCH log stream,
so the strict stdout file was reconstructed from only the deterministic golden
lines.

Earlier completed command-run evidence preserved in this pass:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eigvals_f64_256x256
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eig_f64_256x256
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 rch exec -- cargo run -q -p ft-kernel-cpu --release --example eigvals_golden
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 rch exec -- cargo run --release -q -p ft-kernel-cpu --example eig_timing_probe
```

## Criterion Baseline

Fresh requested-worker rerun:

| Row | Worker | Samples | Time interval |
| --- | --- | ---: | --- |
| `eigvals_f64_256x256` | `vmi1227854` | 100 | `[24.692 ms 25.049 ms 25.415 ms]` |
| `eig_f64_256x256` | `vmi1227854` | 100 | `[73.003 ms 75.979 ms 79.258 ms]` |

The `eig_f64_256x256` Criterion row is substantially slower than both the
predecessor pass-1 row and this pass's `eig_timing_probe` n=256 mean. Treat it
as the current Criterion baseline for Pass 2 same-worker comparison, but re-run
an immediate before/after pair if the next pass implements a source lever.

Earlier same-worker command-run baseline:

| Row | Worker | Samples | Time interval |
| --- | --- | ---: | --- |
| `eigvals_f64_256x256` | `vmi1149989` | 100 | `[33.768 ms 35.625 ms 37.476 ms]` |
| `eig_f64_256x256` | `vmi1149989` | 100 | `[50.173 ms 51.101 ms 52.098 ms]` |

## Strict Golden

Strict stdout SHA-256:

```text
24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725  artifacts/perf/frankentorch-fy8to/pass1_eigvals_golden.strict.stdout
```

This matches the known unchanged strict SHA
`24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.
The earlier `vmi1149989` strict extraction
`pass1_eigvals_golden.strict.extracted.stdout` has the same SHA.

Golden digests:

```text
frankentorch-l9xod eigvals_golden n=64
eigvals_digest=0xbc0583d464b1a211
eig_digest=0xbc0583d464b1a211
frankentorch-l9xod eigvals_golden n=128
eigvals_digest=0x763c4b15d92c4b89
eig_digest=0x763c4b15d92c4b89
frankentorch-l9xod eigvals_golden n=256
eigvals_digest=0x00b87b4996340204
eig_digest=0x00b87b4996340204
```

## Sweep/Deflation/Shift Profile

Fresh `eig_timing_probe` rerun on `vmi1227854` with `threads=10`:

| n | eigvals | eig | eig - eigvals | sweeps | defl1 | defl2 | fallback | exceptional | max width | samples | truncated |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 4.20 ms | 7.48 ms | 3.28 ms | 173 | 28 | 50 | 0 | 0 | 128 | 173 | false |
| 256 | 28.63 ms | 46.30 ms | 17.67 ms | 319 | 14 | 121 | 0 | 0 | 256 | 319 | false |
| 512 | 382.99 ms | 533.77 ms | 150.77 ms | 583 | 10 | 251 | 0 | 0 | 512 | 583 | false |
| 1024 | 2870.73 ms | 4375.77 ms | 1505.04 ms | 1132 | 18 | 503 | 0 | 0 | 1024 | 1132 | false |

First shift samples:

| n | active window | x | y | w | exceptional |
| ---: | --- | ---: | ---: | ---: | --- |
| 128 | `0..127` | `6.690e1` | `5.117e1` | `1.724e0` | false |
| 256 | `0..255` | `1.290e2` | `1.240e2` | `6.569e1` | false |
| 512 | `0..511` | `2.554e2` | `2.588e2` | `5.377e1` | false |
| 1024 | `0..1023` | `5.118e2` | `5.120e2` | `1.282e2` | false |

Earlier `vmi1149989` profile evidence is directionally consistent: n=256
`eigvals=30.15ms`, `eig=51.56ms`, `sweeps=319`, `defl1=14`, `defl2=121`,
`fallback=0`, `exceptional=0`; n=1024 `eigvals=2827.54ms`, `eig=5112.56ms`,
`sweeps=1132`, `defl1=18`, `defl2=503`, `fallback=0`, `exceptional=0`.

## Pass 2 Blockers / Routing Notes

- No source lever was implemented in Pass 1.
- `crates/ft-kernel-cpu/src/lib.rs` was not edited.
- The profile confirms the scalar Francis QR sweep remains the values-path
  floor: n=256 uses 319 sweeps and n=1024 uses 1132 sweeps with no fallback or
  exceptional shifts on this fixture.
- Pass 2 should refine the standalone Schur-window / AED-derived shift-list
  primitive around active-window ownership, Hessenberg invariant checks,
  collision/end handling, and strict scalar fallback accounting. It should not
  repeat range/index micro-cuts or diagnostic-only shift helpers.
