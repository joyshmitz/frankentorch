# frankentorch-5oqum pass 1: two-stage probe harness

## Scope

- Live bead: `frankentorch-5oqum`
- Prior bead check: `frankentorch-34hx9` is closed; the remaining live target is
  the split `5oqum` blocked-panel/two-stage eigensolver work.
- Source change: Criterion benchmark harness only
- Runtime dispatch change: none
- Crate: `ft-kernel-cpu`

## Baseline Evidence Carried Forward

Two current same-worker public baselines are available on `vmi1227854`:

- `artifacts/perf/frankentorch-34hx9/deep_pass1_current_public_and_rank2k.log`
  refreshed the current public pair after the rejected 34hx9 wedge:
  `eigvalsh_f64_256x256` `[7.5447 ms 7.9339 ms 8.2461 ms]`.
- `artifacts/perf/frankentorch-5oqum/pass1_baseline_profile.md` records the
  split-worktree 5oqum baseline:
  `eigvalsh_f64_256x256` `[6.8445 ms 7.0154 ms 7.2518 ms]`.

These rows prove that the public values-only symmetric eigensolver is still a
material target. They do not time the proven two-stage vehicle because the bench
harness previously lacked those rows.

## One Lever

Added only the missing Criterion probes to
`crates/ft-kernel-cpu/benches/linalg_bench.rs`:

- `eigvalsh_two_stage_f64_128x128_b16`
- `eigvalsh_two_stage_f64_256x256_b32`
- `sym_to_banded_f64_128x128_b16`
- `sym_to_banded_f64_256x256_b32`
- `banded_to_tridiag_f64_128x128_b16`
- `banded_to_tridiag_f64_256x256_b32`

The input matrix matches the existing `bench_eigh` symmetric, well-conditioned
matrix family so live and staged rows differ by algorithm, not conditioning.

## RCH Probe Attempt

Combined optimized Criterion command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- 'eigvalsh_f64_256x256|eigvalsh_two_stage_f64_(128x128_b16|256x256_b32)|sym_to_banded_f64_(128x128_b16|256x256_b32)|banded_to_tridiag_f64_(128x128_b16|256x256_b32)' --warm-up-time 1 --measurement-time 3 --sample-size 10
```

Result: no benchmark rows. RCH selected `vmi1227854`, but the stuck detector
canceled the job during optimized `ft-kernel-cpu` compilation:

- job id: `29879662679164834`
- exit code: `130`
- reason: `stuck_detector`
- raw log: `artifacts/perf/frankentorch-5oqum/pass1_two_stage_probe.log`

This is not a performance result. Pass 2 must rerun a narrowed row set, ideally
one staged row at a time, so Criterion output appears before the stuck detector
can kill a large combined compile/bench job.

## Behavior Isomorphism

- Ordering preserved: yes. No runtime operator code or public dispatch changed.
- Tie-breaking unchanged: yes. No eigenvalue/eigenvector sorting code changed.
- Floating-point unchanged: yes. The new code is benchmark-only and does not run
  outside Criterion.
- RNG unchanged: yes. The benchmark input is deterministic arithmetic, matching
  the existing `bench_eigh` matrix family.
- Golden output: not regenerated for this pass because runtime behavior is
  unchanged. Existing public baseline/golden obligations remain the acceptance
  gate for the first runtime lever.

## Gates

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -j 1 -p ft-kernel-cpu --bench linalg_bench`: passed on `vmi1227854`.
- `cargo fmt -p ft-kernel-cpu --check`: passed locally; RCH refused `cargo fmt`
  as a non-compilation command under remote-required mode.
- `ubs crates/ft-kernel-cpu/benches/linalg_bench.rs`: passed with `0` critical
  issues; warnings are pre-existing bench-style `unwrap`, indexing, and format
  allocation inventory.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -j 1 -p ft-kernel-cpu --bench linalg_bench -- -D warnings`: passed on `vmi1227854`.

## Pass 2 Route

Run a narrower RCH optimized bench for the staged rows, starting with:

1. `banded_to_tridiag_f64_128x128_b16`
2. `sym_to_banded_f64_128x128_b16`
3. `eigvalsh_two_stage_f64_128x128_b16`

Then scale to 256 only after the 128 rows emit successfully. If stage 2 is a
large fraction of staged cost, attack the band-packed bulge-chase apply. If
stage 1 dominates, reroute to the DLATRD/SBR blocked panel route.
