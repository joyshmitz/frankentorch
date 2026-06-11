# frankentorch-npxbw Pass 1 - Baseline/Profile

Date: 2026-06-11
Agent: IvoryDeer
Scope: baseline/profile only; no production source edits, no commits, no pushes.

## Commands

- `RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eigvals_f64_256x256 --sample-size 10 --warm-up-time 1 --measurement-time 3`
- `RCH_WORKER=vmi1152480 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eig_f64_256x256 --sample-size 10 --warm-up-time 1 --measurement-time 3`
- `RCH_WORKER=vmi1152480 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never rch exec -- cargo run -p ft-kernel-cpu --example eigvals_golden`

## RCH Status

- `eigvals_f64_256x256`: remote, worker `vmi1152480`, no local fallback.
- `eig_f64_256x256`: remote, worker `vmi1152480`, no local fallback.
- `eigvals_golden`: remote, worker `vmi1152480`, no local fallback.
- Agent Mail build-slot acquisition was unavailable because build slots are disabled on this server.
- Note: the artifact directory also contains RCH-retrieved concurrent `vmi1227854` logs; this pass reports the explicit same-worker `vmi1152480` commands above.

## Baseline Rows

| Benchmark | Criterion row | Median-ish |
| --- | --- | --- |
| `eigvals_f64_256x256` | `[27.044 ms 28.425 ms 29.811 ms]` | `28.425 ms` |
| `eig_f64_256x256` | `[56.971 ms 59.195 ms 61.497 ms]` | `59.195 ms` |

`eig_f64_256x256` emitted Criterion's short-measurement warning and collected 10 samples in an estimated 3.1875s with 55 iterations.

## Golden Output

- Strict golden stdout: `artifacts/perf/frankentorch-npxbw/pass1_eigvals_golden.strict.stdout`
- SHA-256: `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`
- Verification: `sha256sum -c artifacts/perf/frankentorch-npxbw/pass1_eigvals_golden.strict.stdout.sha256` passed.

RCH routed the example's remote stdout through the captured stderr stream; the strict stdout artifact contains only the golden program lines extracted from the successful remote run.

## Profile Diagnosis

The same-worker split shows full `eig` at about `2.08x` the `eigvals` median. `eigvals` still carries the shared non-symmetric Hessenberg/Francis QR wall, while full `eig` adds roughly `30.770 ms` median for eigenvector/back-transform work. The direct small-bulge multishift QR pass should therefore target the shared Francis QR sweep path first, because a real sweep-count or far-update batching win can move both `eigvals` and full `eig`; it should not repeat the rejected `qglh3` AED threshold variants and should stay away from the separate symmetric `eigvalsh` reduction lane.
