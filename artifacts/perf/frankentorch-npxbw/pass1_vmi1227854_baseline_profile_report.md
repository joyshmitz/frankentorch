# frankentorch-npxbw pass 1 baseline/profile

Date: 2026-06-11

Scope: crate-scoped `ft-kernel-cpu` only. No source edits.

## RCH status

- `rch workers probe --all`: 10/10 workers healthy.
- `rch status`: remote-ready, 10/10 healthy, 50/70 slots available; `ovh-a` reported critical storage pressure.
- Advisory Agent Mail build slots were unavailable: `Build slots are disabled. Enable WORKTREES_ENABLED to use this tool.`
- All kept benchmark/profile evidence below used `RCH_REQUIRE_REMOTE=1` and completed remotely on `vmi1227854`. No local fallback was used.

## Commands

```bash
rch workers probe --all
rch status
rch queue
br show frankentorch-npxbw --json
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- eigvals_f64_256x256
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- eig_f64_256x256
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo run --release -q -j 1 -p ft-kernel-cpu --example eig_timing_probe
```

Logs:

- `pass1_baseline_eigvals_f64_256x256_vmi1227854.log`
- `pass1_baseline_eig_f64_256x256_vmi1227854.log`
- `pass1_profile_eig_timing_probe_vmi1227854.log`

## Criterion baseline

| Row | Worker | Samples | Time interval |
| --- | --- | ---: | --- |
| `eigvals_f64_256x256` | `vmi1227854` | 100 | `[24.799 ms 25.258 ms 25.738 ms]` |
| `eig_f64_256x256` | `vmi1227854` | 100 | `[42.861 ms 43.339 ms 43.825 ms]` |

The current live `eigvals` median is slightly faster than the parent fql10 handoff median
`27.645 ms`. The current live `eig` median is much faster than the parent handoff median
`79.126 ms`; treat the parent value as stale or load-sensitive for pass-2 decisions unless
reconfirmed in a same-run A/B.

## Profile/cost split

`eig_timing_probe` on the same worker reported `threads=10`:

| n | eigvals | eig | eig - eigvals |
| ---: | ---: | ---: | ---: |
| 128 | 4.47 ms | 7.19 ms | 2.72 ms |
| 256 | 28.81 ms | 46.06 ms | 17.26 ms |
| 512 | 374.33 ms | 512.41 ms | 138.08 ms |
| 1024 | 2828.59 ms | 4784.97 ms | 1956.38 ms |

Interpretation: the values path remains the primary floor at the target size and grows as the
dominant direct QR residual at larger sizes. This supports routing pass 2 toward direct
small-bulge multishift QR mechanics, not another qglh3 threshold AED variant.

## Bead state

`frankentorch-npxbw` remained `in_progress`, assigned to `IvoryDeer`.
`frankentorch-qglh3` is closed and remains the recorded blocker dependency.
