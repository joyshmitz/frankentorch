# FT-P2C-005 — Rust Implementation Plan + Module Boundary Skeleton

Packet: CPU kernel first-wave semantics  
Subtask: `bd-3v0.16.4`

## 1) Module Boundary Justification

| Crate/module | Ownership | Why this boundary is required | Integration seam |
|---|---|---|---|
| `ft-core` tensor metadata/types | `DType`, `Device`, metadata compatibility contracts | isolates compatibility checks from kernel math and dispatch policy | consumed by kernels, dispatch, and conformance fixtures |
| `ft-kernel-cpu` scalar/binary kernels | scoped add/mul kernel semantics + deterministic error mapping | keeps CPU arithmetic behavior explicit and auditable | selected by dispatcher and validated in packet conformance |
| `ft-dispatch` keyset + routing | dispatch key decoding and kernel decision determinism | centralizes backend/key policy and fail-closed keyset handling | drives packet dispatch evidence fields in logs |
| `ft-conformance` packet FT-P2C-005 suite | fixture execution + strict/hardened comparator envelope | packet-level proof point for kernel/dispatch observable behavior | emits packet differential and e2e artifacts |
| `artifacts/phase2c/FT-P2C-005/*` | contract/risk/threat/evidence docs | provides reproducible governance trail from anchors to parity gates | consumed by packet closure and global validator |

## 2) Low-Risk Implementation Sequence (One Optimization Lever per Step)

| Step | Change scope | Semantic risk strategy | Single optimization lever |
|---|---|---|---|
| `S1` | lock packet fixture + strict/hardened contract rows in conformance | codify behavior before widening implementation surface | fixture parsing once per packet run, reused across modes |
| `S2` | implement/verify dtype-device compatibility fail-closed paths in CPU kernels | preserve deterministic failures before adding broader shape paths | early compatibility short-circuit before arithmetic path |
| `S3` | dispatch decision determinism hardening (`selected_key`/`backend_key`/`kernel`) | enforce replay-stable dispatch tuple prior to differential expansion | cache deterministic keyset decode in run-local map |
| `S4` | add differential/metamorphic/adversarial FT-P2C-005 comparators | gate behavior changes with oracle + adversarial checks | packet-filtered comparator execution to reduce drift-noise |
| `S5` | emit packet-scoped e2e replay/forensics artifacts | require reproducible replay envelope before final closure | packet-filtered e2e matrix path to reduce forensics overhead |
| `S6` | profile + optimize packet hotspots with isomorphism proof | accept only verified behavior-preserving speedups | one lever per change with p50/p95/p99 delta artifact |

## 3) Detailed Test Implementation Plan

### 3.1 Unit/property suite plan

- `ft-kernel-cpu`
  - add/mul scalar correctness across scoped representative values
  - dtype/device incompatibility fail-closed checks
  - deterministic error reason mapping for incompatible inputs
- `ft-dispatch`
  - deterministic keyset -> backend/kernel decision checks
  - invalid/unknown keyset adversarial rejection checks
- `ft-conformance`
  - packet strict/hardened conformance tests for add/mul + broadcast/in-place edge contracts

### 3.2 Differential/metamorphic/adversarial hooks

- differential source artifact:
  - `artifacts/phase2c/conformance/differential_report_v1.json`
- packet slice artifacts (target):
  - `artifacts/phase2c/FT-P2C-005/differential_packet_report_v1.json`
  - `artifacts/phase2c/FT-P2C-005/differential_reconciliation_v1.md`
- adversarial targets:
  - malformed/unknown dispatch keysets
  - dtype/device mismatch coercion probes
  - non-broadcastable shape pair rejection
  - in-place destination compatibility violations

### 3.3 E2E script plan

- packet-scoped e2e command (target):
  - `rch exec -- cargo run -p ft-conformance --bin run_e2e_matrix -- --mode both --packet FT-P2C-005 --output artifacts/phase2c/e2e_forensics/ft-p2c-005.jsonl --print-full-log`
- packet triage/index commands (target):
  - `rch exec -- cargo run -p ft-conformance --bin triage_forensics_failures -- --input artifacts/phase2c/e2e_forensics/ft-p2c-005.jsonl --output artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_005_v1.json --packet FT-P2C-005`
  - `rch exec -- cargo run -p ft-conformance --bin build_failure_forensics_index -- --e2e artifacts/phase2c/e2e_forensics/ft-p2c-005.jsonl --triage artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_005_v1.json --output artifacts/phase2c/e2e_forensics/failure_forensics_index_ft_p2c_005_v1.json`

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

CPU-kernel additions:
- `dispatch_key`
- `selected_kernel`
- `backend_key`
- `input_shape`
- `output_shape`
- `dtype_pair`
- `broadcast_applied`
- `fallback_path`

## 5) Conformance + Benchmark Integration Hooks

Conformance hooks:
- strict/hardened FT-P2C-005 conformance packet suite
- packet differential report + reconciliation artifacts
- packet e2e replay + triage + failure-index artifacts
- packet/global validator (`validate_phase2c_artifacts`)

Benchmark hooks:
- packet microbench for FT-P2C-005 with p50/p95/p99 and mean tracking
- dispatch decision overhead under representative keyset sets
- compatibility-failure path overhead for adversarial fixture corpus

## 6) N/A Cross-Cutting Validation Note

This implementation-plan artifact is docs/planning for subtask D (`bd-3v0.16.4`).  
Execution evidence ownership is explicitly delegated to:
- `bd-3v0.16.5` (unit/property)
- `bd-3v0.16.6` (differential/metamorphic/adversarial)
- `bd-3v0.16.7` (e2e replay/forensics)
- `bd-3v0.16.8` (optimization/isomorphism)
- `bd-3v0.16.9` (final evidence pack)
