# FT-P2C-003 E2E Replay + Forensics Linkage (v1)

## Scope

- e2e log: `artifacts/phase2c/e2e_forensics/ft-p2c-003.jsonl`
- triage summary: `artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_003_v1.json`
- packet failure index: `artifacts/phase2c/e2e_forensics/failure_forensics_index_ft_p2c_003_v1.json`

## Outcomes

- packet filter recorded: `FT-P2C-003`
- total entries: `10` (strict=5, hardened=5)
- failed entries: `0`
- triaged incidents: `0`

Suites exercised:
- `op_schema`

## Deterministic Replay Contract Status

Required fields are present for every log entry:
- `scenario_id`
- `seed`
- `env_fingerprint`
- `artifact_refs`
- `replay_command`
- `reason_code`

## Cross-Evidence Linkage

- unit/property evidence commands are embedded in packet failure index templates.
- differential linkage references:
  - `artifacts/phase2c/conformance/differential_report_v1.json`
  - `artifacts/phase2c/FT-P2C-003/differential_packet_report_v1.json`
  - `artifacts/phase2c/FT-P2C-003/differential_reconciliation_v1.md`
- risk linkage: `artifacts/phase2c/FT-P2C-003/risk_note.md`

## Method-Stack Note

- alien-artifact-coding: replay and forensics linkage emitted as deterministic artifact pair.
- extreme-software-optimization: no optimization lever changed in this bead; behavior-isomorphism preserved by zero-failure packet e2e run.
- RaptorQ-everywhere durability: packet parity sidecar/decode artifacts remain the durability anchor for this packet.
- frankenlibc/frankenfs security-compatibility doctrine: packet-filtered triage confirms strict/hardened fail-closed behavior with no unclassified failures.
