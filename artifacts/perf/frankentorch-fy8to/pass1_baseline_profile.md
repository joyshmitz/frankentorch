# frankentorch-fy8to pass 1 baseline/profile

Date: 2026-06-12T22:13:21Z

Scope: baseline/profile only for the strict-fallback Schur-window shift-list kernel bead. No production source was edited.

## Commands run by this pass

- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eigvals_f64_256x256`
  - Log: `pass1_bench_eigvals_f64_256x256_vmi1227854_attempt1.log`
  - RCH selected `vmi1149989`, so this is remote but not the requested worker.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eig_f64_256x256`
  - Log: `pass1_bench_eig_f64_256x256_vmi1149989_attempt1.log`
  - RCH selected `vmi1149989`.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 rch exec -- cargo run -q -p ft-kernel-cpu --release --example eigvals_golden`
  - Log: `pass1_eigvals_golden.strict.stderr.log`
  - Extracted stdout: `pass1_eigvals_golden.strict.extracted.stdout`
  - SHA file: `pass1_eigvals_golden.strict.extracted.stdout.sha256`
  - RCH selected `vmi1149989`.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 rch exec -- cargo run --release -q -p ft-kernel-cpu --example eig_timing_probe`
  - Log: `pass1_eig_timing_probe_vmi1149989.log`
  - RCH selected `vmi1149989`.

## Baseline rows

Confirmed same-worker remote pair from commands run in this pass:

| Worker | Row | Criterion interval |
|--------|-----|--------------------|
| `vmi1149989` | `eigvals_f64_256x256` | `[33.768 ms 35.625 ms 37.476 ms]` |
| `vmi1149989` | `eig_f64_256x256` | `[50.173 ms 51.101 ms 52.098 ms]` |

Supplemental same-worker `vmi1227854` rows appeared in the same artifact directory from a concurrent crate-scoped RCH run:

| Worker | Row | Criterion interval |
|--------|-----|--------------------|
| `vmi1227854` | `eigvals_f64_256x256` | `[24.692 ms 25.049 ms 25.415 ms]` |
| `vmi1227854` | `eig_f64_256x256` | `[73.003 ms 75.979 ms 79.258 ms]` |

The supplemental `vmi1227854` pair matches the historical anchor worker but was not launched by this pass. Treat it as useful same-worker baseline evidence, not as a before/after decision for a source lever.

## Strict golden

Extracted strict stdout SHA-256:

```text
24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725  artifacts/perf/frankentorch-fy8to/pass1_eigvals_golden.strict.extracted.stdout
```

Printed digests:

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

## Francis profile evidence

`eig_timing_probe` on `vmi1149989` reported `threads=10`.

| n | eigvals | eig | vector machinery | sweeps | defl1 | defl2 | fallback | exceptional | max width | samples | truncated | first shift |
|---|---------|-----|------------------|--------|-------|-------|----------|-------------|-----------|---------|-----------|-------------|
| 128 | 4.57 ms | 8.58 ms | 4.01 ms | 173 | 28 | 50 | 0 | 0 | 128 | 173 | false | `[0..127 x=6.690e1 y=5.117e1 w=1.724e0 exceptional=false]` |
| 256 | 30.15 ms | 51.56 ms | 21.41 ms | 319 | 14 | 121 | 0 | 0 | 256 | 319 | false | `[0..255 x=1.290e2 y=1.240e2 w=6.569e1 exceptional=false]` |
| 512 | 381.63 ms | 562.33 ms | 180.70 ms | 583 | 10 | 251 | 0 | 0 | 512 | 583 | false | `[0..511 x=2.554e2 y=2.588e2 w=5.377e1 exceptional=false]` |
| 1024 | 2827.54 ms | 5112.56 ms | 2285.02 ms | 1132 | 18 | 503 | 0 | 0 | 1024 | 1132 | false | `[0..1023 x=5.118e2 y=5.120e2 w=1.282e2 exceptional=false]` |

Diagnosis: the target fixture still has no fallback or exceptional shifts, records full active-window widths, and remains dominated by the serial Francis QR/eigvals floor. The next pass should design the strict-fallback Schur-window / shift-list primitive around active-window ownership and fallback accounting, not repeat range/index micro-cuts.

## Caveats

- The first requested `vmi1227854` command run by this pass selected `vmi1149989`; the final same-worker pair for commands I launched is therefore `vmi1149989`.
- A concurrent RCH run produced completed `vmi1227854` Criterion rows in `pass1_bench_*_requested_vmi1227854.log`; these are recorded as supplemental.
- A supplemental `pass1_eigvals_golden_requested_vmi1227854.log` was still in progress while this report was drafted, so the strict SHA above uses the completed `vmi1149989` run. The stable extracted stdout uses a distinct filename because RCH retrieval can overwrite files that existed before a concurrent remote sync.
- No source lever was implemented and no behavior changed.
