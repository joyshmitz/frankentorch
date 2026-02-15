# FT-P2C-004 — Security + Compatibility Threat Model

Packet scope: Autograd engine scheduling  
Subtasks: `bd-3v0.15.3` (threat model), with execution linkage to `bd-3v0.15.5`/`.6`/`.7`/`.8`/`.9`

## 1) Threat Model Scope

Protected invariants:
- deterministic dependency scheduling and execution order replay
- strict/hardened mode split without semantic gradient drift
- fail-closed behavior for unknown-node and dependency-underflow paths
- explicit allowlisted hardened fallback for reentrant-depth overflow only

Primary attack/failure surfaces:
- dependency underflow or malformed graph edges causing invalid scheduler state
- nondeterministic ready-queue ordering causing gradient/output drift
- reentrant overflow policy confusion between strict and hardened modes
- telemetry/replay field omissions that make incidents non-reproducible

## 2) Compatibility Envelope and Mode-Split Fail-Closed Rules

| Boundary | Strict mode | Hardened mode | Fail-closed rule |
|---|---|---|---|
| dependency pre-count (`compute_dependencies`) | invalid graph edges fail | same | unknown/malformed graph references are terminal |
| ready-queue scheduling | deterministic tie-break, no fallback | same | no alternate implicit ordering policy |
| dependency decrement (`complete_dependency`) | underflow fails (`DependencyUnderflow`) | same | no counter repair |
| reentrant overflow | hard fail (`ReentrantDepthExceeded`) | bounded fallback with telemetry | fallback only if allowlisted (`autograd.reentrant_depth_bounded_fallback`) |
| replay/evidence envelope | all fields required | all fields required | missing deterministic replay fields are reliability-gate failures |

## 3) Packet-Specific Abuse Classes and Defensive Controls

| Threat ID | Abuse class | Attack surface | Impact if unmitigated | Defensive control | Strict/Hardened expectation | Unit/property fixture mapping | Failure-injection e2e scenario seed(s) |
|---|---|---|---|---|---|---|---|
| `THR-401` | dependency underflow corruption | scheduler decrement path | invalid node release ordering and gradient drift | fail-closed `DependencyUnderflow` error contract | strict=fail, hardened=fail | candidate `ft_autograd` underflow regression assertion; `ft_autograd::dependency_scheduler_waits_for_all_children` | candidate seeds: `autograd_scheduler/strict:dependency_underflow_probe`=`8420101970047811095`, `autograd_scheduler/hardened:dependency_underflow_probe`=`5298414977992276633` |
| `THR-402` | ordering nondeterminism | ready queue tie-break path | replay instability + parity drift | deterministic queue ordering with stable NodeId tie-break | strict=deterministic, hardened=deterministic | `ft_autograd::composite_graph_gradient_is_deterministic`, `ft_conformance::strict_scheduler_conformance_is_green` | `autograd_scheduler/strict:composite_scheduler_dac`=`7059225966474559304`, `autograd_scheduler/hardened:composite_scheduler_dac`=`12527830154622575501` |
| `THR-403` | reentrant overflow policy bypass | reentrant-depth checks | silent policy drift or scheduler corruption | strict hard fail + hardened bounded allowlisted fallback | strict=fail-closed, hardened=bounded fallback | `ft_autograd::strict_mode_reentrant_depth_overflow_fails`, `ft_autograd::hardened_mode_reentrant_depth_overflow_fallbacks` | `autograd_scheduler/strict:scheduler_chain`=`12856897130756503459`, `autograd_scheduler/hardened:scheduler_chain`=`4056182831483653040` |
| `THR-404` | allowlist abuse | hardened policy comparator path | unbounded behavior-altering repair | require allowlist ID + explicit telemetry evidence | strict=no drift, hardened=single bounded drift id only | differential policy comparator checks in `ft-conformance` | same packet scenario pair as `THR-402` |
| `THR-405` | replay evidence omission | scheduler e2e logging envelope | incident cannot be replayed/audited | mandatory replay schema fields + artifact refs | same in both modes | logging crosswalk + e2e matrix schema checks | packet replay seeds under `autograd_scheduler/*` |
| `THR-406` | out-of-scope concurrency assumptions | multithread/device worker semantics | false confidence from scoped model | explicit deferred-gap declaration | strict/hardened both treat as deferred parity edge | deferred to later packet chain | candidate seeds: `autograd_scheduler/strict:multithread_queue_interleave_gap`=`11048271943701054945`, `autograd_scheduler/hardened:multithread_queue_interleave_gap`=`1673982333076546041` |

## 4) Mandatory Forensic Logging + Replay Artifacts for Incidents

For every scheduler incident, logs must include:
- `suite_id`
- `scenario_id`
- `packet_id`
- `mode`
- `seed`
- `execution_order`
- `queue_pushes`
- `queue_pops`
- `max_queue_len`
- `dependency_snapshot`
- `reentrant_depth`
- `reentrant_guard_triggered`
- `hardened_fallback_used`
- `artifact_refs`
- `replay_command`
- `env_fingerprint`

Required artifact linkage chain:
1. packet e2e log (`artifacts/phase2c/e2e_forensics/ft-p2c-004.jsonl`)
2. packet failure triage (`artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_004_v1.json`)
3. packet failure forensics index (`artifacts/phase2c/e2e_forensics/failure_forensics_index_ft_p2c_004_v1.json`)
4. differential source report (`artifacts/phase2c/conformance/differential_report_v1.json`)
5. packet differential slice (`artifacts/phase2c/FT-P2C-004/differential_packet_report_v1.json`)
6. packet differential reconciliation (`artifacts/phase2c/FT-P2C-004/differential_reconciliation_v1.md`)
7. packet contract anchor (`artifacts/phase2c/FT-P2C-004/contract_table.md`)

## 5) Residual Risks and Deferred Controls

Residual risks:
- packet scope is intentionally single-threaded CPU scheduling; full worker-thread/device queue semantics are deferred.
- only one hardened allowlisted policy drift is permitted; additional drift IDs are release-blocking.

Deferred controls and ownership:
- `bd-3v0.15.5`: unit/property expansion for underflow and shared-subgraph adversarial edges.
- `bd-3v0.15.6`: differential/metamorphic/adversarial reconciliation including allowlist drift controls.
- `bd-3v0.15.7`: packet e2e replay/forensics pipeline artifacts (`ft-p2c-004` slice).
- `bd-3v0.15.8`: scheduler tail-latency optimization with behavior-preserving proof.
- `bd-3v0.15.9`: final evidence pack closure (parity + RaptorQ sidecars + decode proof).

## 6) N/A Cross-Cutting Validation Note

This risk note is docs/planning for packet subtask C (`bd-3v0.15.3`).
Execution evidence ownership:
- unit/property: `bd-3v0.15.5`
- differential/adversarial: `bd-3v0.15.6`
- e2e/replay forensics: `bd-3v0.15.7`
