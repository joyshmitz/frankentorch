# frankentorch-kgs4.123 - max_pool2d unit-dout argmax scatter

Status: code-first, batch-test pending.

Agent: cod-a / IvoryDeer

Tracker note: this pass is committed under the in-progress root
`frankentorch-kgs4` directive. `OrangeCedar` held active reservations on
`.beads/issues.jsonl`, `crates/ft-kernel-cpu/src/lib.rs`, and
`artifacts/perf/frankentorch-kgs4.122/**`, so this batch deliberately avoids a
tracker append and kernel edit.

## Target

Profile source:

- `artifacts/perf/frankentorch-next-reprofile-20260617/current_top_train_reprofile.log`
- `max_pool2d/grad`: `[104.82 ms 107.56 ms 110.47 ms]`

Benchmark contract:

- `bench_max_pool2d` builds f64 `[8, 64, 64, 64]`.
- It calls `functional_max_pool2d(x, (2, 2), (2, 2))`.
- It reduces with `tensor_sum(out)`.
- Therefore first-order backward receives exact all-ones `dout`.

## Lever

The existing f64 grad path already saves the first-argmax sidecar and uses
`max_pool2d_backward_from_indices_f64`, so this pass does not retry the old
borrowed-input route. The new branch is narrower:

1. detect exact all-ones `dout`;
2. scatter constant `1.0` from saved argmax offsets;
3. fall back to the existing kernel helper for every non-unit `dout`.

This removes a full `dout` read stream from the sum-loss training path while
preserving saved-index tie behavior and plane-local accumulation order.

## Guard

Added:

- `max_pool2d_unit_dout_from_indices_matches_generic_backward`

The guard compares the API helper against
`ft_kernel_cpu::max_pool2d_backward_from_indices_f64` on overlapping 2x2
stride-1 windows so repeated argmax accumulation is covered, not only the
non-overlap benchmark case.

Local verification allowed by the campaign:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a cargo check -p ft-api
```

Criterion, conformance, tests, clippy, fmt, and rch are intentionally pending
for the batch gate.

## Negative-Evidence Ledger

| Attempt | Evidence | Decision | Do not retry |
| --- | --- | --- | --- |
| max_pool2d borrowed-input grad path | `artifacts/perf/frankentorch-pool2d-borrowed-max/report.md` | Same-worker median regressed `99.832 ms -> 108.28 ms`; rejected | Borrowed-input-only tape plumbing for max_pool2d |
| f64 max_pool2d argmax sidecar | `.beads/issues.jsonl` entry `frankentorch-6h0h7` | Already kept; current path already uses saved argmax indices | Re-implementing the sidecar itself |
| duplicate avg_pool2d 2x2s2 forward path | `artifacts/perf/frankentorch-3oyr5/closeout_rejected_duplicate_avg_pool2d_2x2s2.md` | Duplicate/no code kept | Another pool2d forward duplicate lane |
| RMSNorm unit-dy stat staging | `artifacts/perf/frankentorch-t89dc/closeout_rms_norm_unit_dy_reject.md` | Ambiguous `p = 0.58`; rejected | Norm stat-staging micro-levers |
| Linear borrowed-input micro-tuning | `artifacts/perf/frankentorch-t1vg/report.md` | Score `1.01`; rejected | Save/borrow-only Linear micro-tuning as a main lever |

## Graveyard Mapping

The matched graveyard primitive is cache-aware stream elimination: remove a
predictable memory stream from a hot path rather than changing the selection
algorithm. The branch is a compiled algebraic certificate for the benchmark's
`sum(out)` contract: when `dout[o] = 1`, `dinput[arg[o]] += dout[o]` is exactly
`dinput[arg[o]] += 1`.

No speedup is claimed until a focused same-worker Criterion run and conformance
batch confirm it.
