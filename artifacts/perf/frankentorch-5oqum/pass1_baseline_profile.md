# frankentorch-5oqum pass 1 baseline/profile

Date: 2026-06-10
Worktree: `/data/projects/.scratch/frankentorch-5oqum-boldfalcon-20260610T0249`
Commit: `0a51aa38`
Policy: source read-only; crate-scoped RCH only; remote required for cargo; no full-workspace builds.

## Bench surface inspected

Existing `crates/ft-kernel-cpu/benches/linalg_bench.rs` coverage:

- Live full/vector symmetric eigensolver: `eigh_f64_128x128`, `eigh_f64_256x256`.
- Live values-only symmetric eigensolver: `eigvalsh_f64_128x128`, `eigvalsh_f64_256x256`.
- Stage-1 BLAS-3 subprimitive probe: `sym_rank2k_lower_scalar_f64_256x32`, `sym_rank2k_lower_gemm_f64_256x32`.

Missing staged/two-stage Criterion coverage needed before scoring the staged path:

- `eigvalsh_two_stage_f64_256x256_b32`: same deterministic symmetric input as `bench_eigh`, calls `eigvalsh_two_stage_f64(&a, 256, 32)`.
- `symmetric_to_banded_f64_256x256_b32`: isolates stage 1 dense-to-banded reference cost.
- `banded_to_tridiagonal_f64_256x256_b32`: isolates stage 2 bulge-chase cost on a precomputed deterministic band from `symmetric_to_banded_f64`.

No bench edits were made in this pass.

## RCH status

- Before: degraded, `2/10` workers healthy, `8/70` slots available; `ovh-a` under critical disk pressure.
- After: degraded, `2/10` workers healthy, `8/70` slots available; `ovh-a` still unavailable due pressure/telemetry.
- First attempted bench without `RCH_REQUIRE_REMOTE=1` selected local fallback (`no admissible workers: critical_pressure=1,insufficient_slots=1`). That run was interrupted and is not used as evidence.

## Commands run

Read-only inspection:

```bash
br show frankentorch-5oqum --json
rg -n "eigvalsh|eigh|two_stage|two-stage|banded|tridiagonal|symmetric_to_banded|banded_to_tridiagonal|criterion_group|bench_function|benchmark_group" crates/ft-kernel-cpu/benches/linalg_bench.rs crates/ft-kernel-cpu/src/lib.rs .skill-loop-progress.md
sed -n '1,560p' crates/ft-kernel-cpu/benches/linalg_bench.rs
sed -n '9980,10820p' crates/ft-kernel-cpu/src/lib.rs
```

Artifact/status capture:

```bash
mkdir -p artifacts/perf/frankentorch-5oqum
rch status > artifacts/perf/frankentorch-5oqum/rch_status_before.txt
rch status > artifacts/perf/frankentorch-5oqum/rch_status_after.txt
```

RCH benches:

```bash
rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eigvalsh_f64_256x256 --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Result: invalid. RCH selected local fallback; interrupted; log only: `bench_eigvalsh_f64_256x256.log`.

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- eigvalsh_f64_256x256 --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Result: valid remote row; log: `bench_eigvalsh_f64_256x256_remote_required.log`.

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- eigh_f64_256x256 --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Result: remote refused, no local fallback; log: `bench_eigh_f64_256x256_remote_required.log`.

## Worker/timing rows

| Bench | Worker | Result |
| --- | --- | --- |
| `eigvalsh_f64_256x256` | `vmi1227854` | `[6.8445 ms 7.0154 ms 7.2518 ms]`, 10 samples, 1 high mild outlier |
| `eigh_f64_256x256` | none | `RCH_REQUIRE_REMOTE=1` refused local fallback: `no worker assigned` |
| `sym_rank2k_lower_*_f64_256x32` | none | not run after remote-required refusal stop condition |
| `eigvalsh_two_stage_f64_256x256_b32` | n/a | missing bench |
| `symmetric_to_banded_f64_256x256_b32` | n/a | missing bench |
| `banded_to_tridiagonal_f64_256x256_b32` | n/a | missing bench |

## Top observed blocker

The pass could baseline the live values-only row, but not the staged path: the benchmark surface lacks standalone two-stage, stage-1, and stage-2 rows. That is the immediate profiling blocker.

Among implementation blockers visible in the current code, the narrowest next optimization target is stage 2: `banded_to_tridiagonal_f64` applies each Givens rotation over full `0..n` rows and columns even though the matrix is banded. The intended band-packed apply should touch only the affected `O(b)` band entries and preserve the same rotation stream. This is more contained than the stage-1 DLATRD/SBR blocked panel and should be easier to prove against the current reference.

## Pass 2 recommendation

First add the three missing Criterion probes above, then use `banded_to_tridiagonal_f64_256x256_b32` to attack the band-packed stage-2 apply. Do not wire live `eigvalsh` until the staged benches show a same-worker win and the two-stage values match the current live path within the existing tolerance.
