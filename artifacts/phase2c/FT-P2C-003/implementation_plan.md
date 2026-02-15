# FT-P2C-003 — Rust Implementation Plan + Module Boundary Skeleton

Packet: Op schema ingestion  
Subtask: `bd-3v0.14.4`

## 1) Module Boundary Justification

| Crate/module | Ownership | Why this boundary is required | Integration seam |
|---|---|---|---|
| `ft-dispatch::schema` (planned) | schema parse/normalize model for scoped operator rows | isolates schema canonicalization from runtime kernel execution policy | consumed by dispatcher route selection and conformance harness |
| `ft-dispatch::schema_registry` (planned) | maps normalized `(op, overload)` to dispatch metadata and scoped kernel route hints | centralizes fail-closed compatibility checks for unknown/incompatible schema metadata | feeds `dispatch_scalar_binary[_with_keyset]` and future multi-op routing |
| `ft-dispatch::DispatchKeySet` (existing) | keyset validation and priority/backends | reused as fail-closed guardrail for schema-derived dispatch metadata | schema registry emits keysets for validation + routing |
| `ft-kernel-cpu` | concrete CPU kernel endpoints (`add_scalar`, `mul_scalar`) | keeps execution kernels isolated from schema parsing and policy | invoked through dispatch layer after schema-derived route resolution |
| `ft-api::FrankenTorchSession` | user-facing op invocation + evidence logging | ensures schema-derived route decisions are captured in deterministic session evidence | logs `op_name`, `overload_name`, route keys, and replay fields |
| `ft-conformance` | fixture-driven schema + dispatch parity/differential/e2e evidence | single source of truth for packet parity gates and replay artifacts | consumes schema fixtures and emits packet artifacts/logs |

## 2) Low-Risk Implementation Sequence (One Optimization Lever per Step)

| Step | Change scope | Semantic risk strategy | Single optimization lever |
|---|---|---|---|
| `S1` | introduce schema DTO + parser facade in `ft-dispatch` | start with strict fail-closed parser behavior before adding routing logic | cache normalized `(op, overload)` key per parsed schema row |
| `S2` | add schema registry + dispatch metadata validation | validate schema-to-keyset mapping independently before kernel route execution | precompute validated keyset bits for repeated lookups |
| `S3` | wire schema-derived routing to scoped CPU/autograd paths | preserve existing FT-P2C-002 route behavior while adding schema gate | avoid duplicate keyset validation in already-validated schema paths |
| `S4` | integrate conformance schema fixtures + differential checks | classify strict/hardened parity before adding adversarial expansion | parse fixture schema strings once per case and reuse digest |
| `S5` | add e2e replay/forensics schema event fields | enforce deterministic scenario IDs and replay commands | single-pass JSONL event emission for schema scenario slices |

## 3) Detailed Test Implementation Plan

### 3.1 Unit/property suite plan

- `ft-dispatch` (planned)
  - `schema_row_parse_round_trips_add_tensor_signature`
  - `operator_name_parse_preserves_overload_token`
  - `base_operator_name_parse_inplace_suffix_contract`
  - `schema_out_variant_requires_mutable_out_alias`
  - `schema_parser_rejects_malformed_tokens`
  - `schema_dispatch_keyset_rejects_unknown_backend_key`
- `ft-api` (planned)
  - session evidence contains schema identity + dispatch-route fields

### 3.2 Differential/metamorphic/adversarial hooks

- differential packet evidence source (existing global report path):
  - `artifacts/phase2c/conformance/differential_report_v1.json`
- schema-specific adversarial candidates:
  - malformed schema rows
  - invalid overload token names
  - dispatch metadata incompatible with scoped keyset policy
- hardened allowlist posture:
  - diagnostics context may vary; accepted semantics cannot vary

### 3.3 E2E script plan

- packet-scoped e2e run command (planned in later bead):
  - `cargo run -p ft-conformance --bin run_e2e_matrix -- --mode both --packet FT-P2C-003 --output artifacts/phase2c/e2e_forensics/ft-p2c-003.jsonl`
- replay commands are sourced from scenario log entries and triaged via failure-index tooling.

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

Schema-ingestion-specific instrumentation points:
- `op_name`
- `overload_name`
- `schema_digest`
- `dispatch_keyset_bits`
- parser classification (`name_only` vs `full_schema`)
- schema route decision reason (`schema_validation_pass`, `schema_rejected`)

## 5) Conformance + Benchmark Integration Hooks

Conformance hooks (planned/extended):
- schema-focused conformance entrypoint in `ft-conformance`
- differential comparator integration for schema parse + route expectations
- packet-filtered e2e forensics export for `FT-P2C-003`
- failure triage + forensics index pipeline

Benchmark hooks:
- schema parse latency (p50/p95/p99) for scoped fixture corpus
- schema-registry lookup latency by `(op, overload)`
- dispatch-route latency for schema-derived decisions vs baseline route path

## 6) N/A Cross-Cutting Validation Note

This implementation-plan artifact is docs/planning only for subtask D.
Execution evidence is deferred to:
- `bd-3v0.14.5` (unit/property with detailed logs)
- `bd-3v0.14.6` (differential/metamorphic/adversarial)
- `bd-3v0.14.7` (e2e scenarios + replay/forensics logs)
