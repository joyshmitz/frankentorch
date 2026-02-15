# FT-P2C-004 — Contract Table + Strict/Hardened Invariant Spec

Packet scope: Autograd engine scheduling (single-threaded deterministic scheduler)

## 1) Scheduler Interface Contracts (Scoped)

| Contract ID | Input | Output | Error Contract | Deterministic Invariant |
|---|---|---|---|---|
| `AUTOGRAD-SCHED-001` | rooted tape graph (`NodeId root`) | reachability mask | `UnknownNode` if root or edge target is invalid | reachability mask is identical for identical graph topology |
| `AUTOGRAD-SCHED-002` | reachable mask + tape graph | dependency counters | `UnknownNode` if dependency edge points outside tape | dependency count equals number of reachable downstream consumers |
| `AUTOGRAD-SCHED-003` | dependency counters + ready queue | execution order vector | `DependencyUnderflow` on invalid decrement | ready-queue tie-break remains stable (`higher NodeId first`) |
| `AUTOGRAD-SCHED-004` | rooted graph + output seed gradient | gradient map (`BackwardReport.gradients`) | `UnknownNode`, `DependencyUnderflow` | gradients are replay-stable for identical graph + options |
| `AUTOGRAD-SCHED-005` | overflowed reentrant depth in strict mode | terminal failure | `ReentrantDepthExceeded` (fail-closed) | no recovery path is permitted in strict mode |
| `AUTOGRAD-SCHED-006` | overflowed reentrant depth in hardened mode | bounded fallback report | no hard error when bounded fallback is allowed | fallback is explicit and telemetry-marked |
| `AUTOGRAD-SCHED-007` | scheduler run completion | telemetry envelope | none | telemetry fields are fully populated and replayable |
| `AUTOGRAD-SCHED-008` | hardened fallback occurrence | policy comparator output | non-allowlisted drift is release-blocking | only allowlisted drift ID `autograd.reentrant_depth_bounded_fallback` is permitted |

## 2) Strict/Hardened Invariant Spec

| Invariant ID | Strict Mode | Hardened Mode | Allowlist / Policy Link |
|---|---|---|---|
| `INV-REENTRANT-DEPTH` | `current_reentrant_depth > max_reentrant_depth` must fail with `ReentrantDepthExceeded` | depth is clamped to bound; execution continues | `artifacts/phase2c/HARDENED_DEVIATION_ALLOWLIST_V1.json` (`autograd.reentrant_depth_bounded_fallback`) |
| `INV-QUEUE-ORDER` | deterministic order for equal-ready sets | same | no divergence permitted |
| `INV-DEPENDENCY-MONOTONE` | counters must never underflow | same | no divergence permitted |
| `INV-GRADIENT-REPLAY` | identical graph/options produce identical gradients | same | no divergence permitted |
| `INV-TELEMETRY-COMPLETENESS` | all scheduler telemetry fields required | same | missing fields are reliability-gate violations |
| `INV-HARDENED-FLAGGING` | `reentrant_guard_triggered=false`, `hardened_fallback_used=false` for non-overflow paths | fallback path must set `reentrant_guard_triggered=true` and `hardened_fallback_used=true` | hardened bounded fallback must remain explicit in evidence logs |

## 3) Error Contract Matrix

| Error Type | Trigger | Strict Behavior | Hardened Behavior |
|---|---|---|---|
| `UnknownNode` | root/edge references missing node | fail-closed | fail-closed |
| `DependencyUnderflow` | dependency decrement below zero | fail-closed | fail-closed |
| `ReentrantDepthExceeded` | reentrant overflow | fail-closed error | converted to bounded fallback report with telemetry |

## 4) Deterministic Telemetry Contract

`BackwardReport.telemetry` required fields:
- `execution_order`
- `queue_pushes`
- `queue_pops`
- `max_queue_len`
- `dependency_snapshot`
- `reentrant_depth`
- `reentrant_guard_triggered`
- `hardened_fallback_used`

Determinism requirements:
1. identical graph + identical options => byte-identical telemetry sequence values.
2. strict runs with no overflow must never set hardened fallback flags.
3. hardened overflow path must preserve gradient correctness while recording explicit policy drift evidence.
