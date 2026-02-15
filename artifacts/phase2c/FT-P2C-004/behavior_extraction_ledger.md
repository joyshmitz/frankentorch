# FT-P2C-004 — Behavior Extraction Ledger

Packet: Autograd engine scheduling  
Legacy anchor map: `artifacts/phase2c/FT-P2C-004/legacy_anchor_map.md`

## Behavior Families (Nominal, Edge, Adversarial)

| Behavior ID | Path class | Legacy anchor family | Strict expectation | Hardened expectation | Candidate unit/property assertions | E2E scenario seed(s) |
|---|---|---|---|---|---|---|
| `FTP2C004-B01` | nominal | `Engine::compute_dependencies`, `Engine::thread_main`, `NodeTask` | deterministic gradient outputs and stable execution order for equivalent graph | same semantic outputs/order | `ft_autograd::composite_graph_gradient_is_deterministic`, `ft_conformance::strict_scheduler_conformance_is_green` | `autograd_scheduler/strict:composite_scheduler_dac`=`7059225966474559304`, `autograd_scheduler/hardened:composite_scheduler_dac`=`12527830154622575501` |
| `FTP2C004-B02` | nominal | dependency pre-count + ready-queue release (`compute_dependencies`, `ReadyQueue::pop`) | parent node executes only after all children complete | same | `ft_autograd::dependency_scheduler_waits_for_all_children` | candidate seeds: `autograd_scheduler/strict:dependency_fanin_gate`=`1308719204661099412`, `autograd_scheduler/hardened:dependency_fanin_gate`=`14919089296112050870` |
| `FTP2C004-B03` | edge (mode split) | reentrant depth boundary (`MAX_DEPTH`, `GraphTask::reentrant_depth`) | overflow must fail closed with `ReentrantDepthExceeded` | bounded fallback allowed only with explicit telemetry | `ft_autograd::strict_mode_reentrant_depth_overflow_fails`, `ft_autograd::hardened_mode_reentrant_depth_overflow_fallbacks` | `autograd_scheduler/strict:scheduler_chain`=`12856897130756503459`, `autograd_scheduler/hardened:scheduler_chain`=`4056182831483653040` |
| `FTP2C004-B04` | edge (allowlist) | policy comparator in conformance output | no policy drift allowed | exactly one bounded allowlisted drift: `autograd.reentrant_depth_bounded_fallback` | differential policy comparator assertions in `ft-conformance` (`reason_code=reentrant_policy_match` / allowlisted drift) | `autograd_scheduler/strict:composite_scheduler_dac`=`7059225966474559304`, `autograd_scheduler/hardened:composite_scheduler_dac`=`12527830154622575501` |
| `FTP2C004-B05` | adversarial | dependency decrement path (`complete_dependency`) | dependency underflow is terminal (`DependencyUnderflow`) | same fail-closed behavior | candidate assertion around invalid dependency counter transitions in `ft-autograd` | candidate seeds: `autograd_scheduler/strict:dependency_underflow_probe`=`8420101970047811095`, `autograd_scheduler/hardened:dependency_underflow_probe`=`5298414977992276633` |
| `FTP2C004-B06` | deferred parity edge | full worker-thread/reentrant queue orchestration (`reentrant_thread_init`, non-reentrant thread counters) | out of packet scope; unsupported multithreaded paths must not silently succeed | same | deferred to future packet chain (thread/device scheduling expansion) | candidate seeds: `autograd_scheduler/strict:multithread_queue_interleave_gap`=`11048271943701054945`, `autograd_scheduler/hardened:multithread_queue_interleave_gap`=`1673982333076546041` |

## Logging Field Expectations by Behavior Family

Mandatory deterministic replay fields (all scheduler behavior families):
- `suite_id`
- `scenario_id`
- `packet_id`
- `mode`
- `seed`
- `env_fingerprint`
- `artifact_refs`
- `replay_command`
- `outcome`
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

Anchors:
- `crates/ft-autograd/src/lib.rs`
- `crates/ft-conformance/src/lib.rs`
- `artifacts/phase2c/UNIT_E2E_LOGGING_CROSSWALK_V1.json`
- `artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl`

## N/A Cross-Cutting Validation Note

This ledger is docs/planning only for packet subtask A (`bd-3v0.15.1`).
Execution evidence ownership is carried by downstream packet beads:
- unit/property: `bd-3v0.15.5`
- differential/adversarial: `bd-3v0.15.6`
- e2e/logging: `bd-3v0.15.7`
