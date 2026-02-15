# FT-P2C-004 — Rust Implementation Plan + Module Boundary Skeleton

Packet: Autograd engine scheduling  
Subtask: `bd-3v0.15.4`

## 1) Module Boundary Justification

| Crate/module | Ownership | Why this boundary is required | Integration seam |
|---|---|---|---|
| `ft-autograd::Tape` | graph construction and backward scheduler core | isolates dependency counting, ready queue behavior, and gradient propagation invariants | consumed by `ft-api` and conformance harness |
| `ft-autograd::BackwardOptions` | strict/hardened mode + reentrant policy envelope | central place to encode policy splits without contaminating scheduler internals | passed from `ft-api` and conformance scenarios |
| `ft-autograd::SchedulerTelemetry` | deterministic execution evidence | separates behavioral correctness from evidence/logging transport concerns | surfaced via `BackwardReport` to API/runtime/conformance |
| `ft-api::FrankenTorchSession` | user-facing backward invocation | enforces mode-aware invocation contract and report forwarding | translates user calls into `Tape::backward_with_options` |
| `ft-conformance` scheduler suite | fixture-driven parity/differential/e2e checks | packet-level behavioral oracle and replay artifacts | consumes `autograd_scheduler_cases.json`, emits packet evidence |
| `artifacts/phase2c/FT-P2C-004/*` | packet contracts/risk/evidence docs | lock-step documentation + evidence traceability for closure gates | consumed by `validate_phase2c_artifacts` and final packet evidence pack |

## 2) Low-Risk Implementation Sequence (One Optimization Lever per Step)

| Step | Change scope | Semantic risk strategy | Single optimization lever |
|---|---|---|---|
| `S1` | deterministic dependency pre-count and ready queue model in `ft-autograd` | enforce fail-closed behavior before adding policy split complexity | preallocate dependency/gradient buffers per tape size |
| `S2` | strict/hardened reentrant-depth policy branch in `BackwardOptions` path | prove strict failure + hardened bounded fallback with explicit tests | avoid extra branching in nominal non-overflow fast path |
| `S3` | conformance scheduler fixtures + policy comparators | add parity checks before adversarial expansions | share fixture parsing across strict/hardened runs |
| `S4` | e2e/replay-forensics scheduler scenario integration | ensure deterministic replay schema and seeds before closing packet | packet filter short-circuit to reduce e2e matrix overhead |
| `S5` | optimization/isomorphism pass for scheduler tails | require equivalence checks before accepting perf deltas | queue bookkeeping compaction with invariant-preserving proof |

## 3) Detailed Test Implementation Plan

### 3.1 Unit/property suite plan

- `ft-autograd`
  - `composite_graph_gradient_is_deterministic`
  - `dependency_scheduler_waits_for_all_children`
  - `strict_mode_reentrant_depth_overflow_fails`
  - `hardened_mode_reentrant_depth_overflow_fallbacks`
  - candidate: dependency underflow adversarial fixture

### 3.2 Differential/metamorphic/adversarial hooks

- differential source report:
  - `artifacts/phase2c/conformance/differential_report_v1.json`
- scheduler comparators:
  - output gradients (`abs_tol`)
  - execution-order equivalence
  - policy comparator (`reentrant_policy_match`, allowlisted hardened fallback)
- adversarial targets:
  - dependency underflow
  - malformed graph edge references (`UnknownNode`)
  - reentrant overflow depth probes

### 3.3 E2E script plan

- packet-scoped e2e command:
  - `rch exec -- cargo run -p ft-conformance --bin run_e2e_matrix -- --mode both --packet FT-P2C-004 --output artifacts/phase2c/e2e_forensics/ft-p2c-004.jsonl`
- deterministic scenario IDs:
  - `autograd_scheduler/strict:composite_scheduler_dac`
  - `autograd_scheduler/hardened:composite_scheduler_dac`

## 4) Structured Logging Instrumentation Points

Required packet fields:
- `scenario_id`
- `packet_id`
- `mode`
- `seed`
- `env_fingerprint`
- `artifact_refs`
- `replay_command`
- `reason_code`

Scheduler-specific additions:
- `execution_order`
- `queue_pushes`
- `queue_pops`
- `max_queue_len`
- `dependency_snapshot`
- `reentrant_depth`
- `reentrant_guard_triggered`
- `hardened_fallback_used`

## 5) Conformance + Benchmark Integration Hooks

Conformance hooks:
- strict/hardened scheduler conformance tests
- differential scheduler comparator slice for `FT-P2C-004`
- packet-filtered e2e forensics slice + crash triage + failure index

Benchmark hooks:
- scheduler `p50/p95/p99` latency under packet fixture loops
- queue operation counts (`queue_pushes`, `queue_pops`) and max depth tracking
- replay-stability hash check for execution order vectors

## 6) N/A Cross-Cutting Validation Note

This implementation-plan artifact is docs/planning only for subtask D.
Execution evidence is owned by:
- `bd-3v0.15.5` (unit/property + detailed structured logs)
- `bd-3v0.15.6` (differential/metamorphic/adversarial)
- `bd-3v0.15.7` (e2e replay/forensics logs)
