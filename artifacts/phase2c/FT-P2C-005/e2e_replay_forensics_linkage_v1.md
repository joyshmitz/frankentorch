# FT-P2C-005 E2E Replay + Forensics Linkage (v1)

## Scope

- e2e log: `artifacts/phase2c/e2e_forensics/ft-p2c-005.jsonl`
- triage summary: `artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_005_v1.json`
- packet failure index: `artifacts/phase2c/e2e_forensics/failure_forensics_index_ft_p2c_005_v1.json`

## Outcomes

- packet filter recorded: `FT-P2C-005`
- total entries: `48` (strict=24, hardened=24)
- failed entries: `0`
- triaged incidents: `0`

Suites exercised:
- `scalar_dac`
- `tensor_meta`
- `dispatch_key`

## Deterministic Replay Contract Status

Required fields are present for every log entry:
- `scenario_id`
- `seed`
- `env_fingerprint`
- `artifact_refs`
- `replay_command`
- `reason_code`

## Cross-Evidence Linkage

- unit/property evidence commands are embedded in packet linkage and failure-index templates.
- differential linkage references:
  - `artifacts/phase2c/conformance/differential_report_v1.json`
  - `artifacts/phase2c/FT-P2C-005/differential_packet_report_v1.json`
  - `artifacts/phase2c/FT-P2C-005/differential_reconciliation_v1.md`
- risk linkage: `artifacts/phase2c/FT-P2C-005/threat_model.md`

## Method-Stack Note

- alien-artifact-coding: packet-scoped replay + forensics artifacts emitted with deterministic scenario and seed lineage.
- extreme-software-optimization: no optimization lever changed; this bead validates e2e reproducibility and forensics completeness only.
- RaptorQ-everywhere durability: durability sidecars remain anchored in packet parity artifacts and final evidence-pack closure.
- security/compatibility doctrine: strict+hardened packet e2e runs are replayable and classify to zero incidents under current fixture envelope.
