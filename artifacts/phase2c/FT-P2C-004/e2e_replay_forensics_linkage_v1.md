# FT-P2C-004 E2E Replay + Forensics Linkage (v1)

## Scope

- e2e log: `artifacts/phase2c/e2e_forensics/ft-p2c-004.jsonl`
- triage summary: `artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_004_v1.json`
- packet failure index: `artifacts/phase2c/e2e_forensics/failure_forensics_index_ft_p2c_004_v1.json`

## Outcomes

- packet filter recorded: `FT-P2C-004`
- total entries: `6` (strict=3, hardened=3)
- failed entries: `0`
- triaged incidents: `0`

Suites exercised:
- `autograd_scheduler`

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
  - `artifacts/phase2c/FT-P2C-004/differential_packet_report_v1.json`
  - `artifacts/phase2c/FT-P2C-004/differential_reconciliation_v1.md`
- risk linkage: `artifacts/phase2c/FT-P2C-004/risk_note.md`

## Method-Stack Note

- alien-artifact-coding: replay + forensics linkage emitted as deterministic packet-scoped artifacts.
- extreme-software-optimization: no optimization lever changed in this bead; behavior-isomorphism preserved with zero-failure packet e2e run.
- RaptorQ-everywhere durability: packet parity sidecar/decode artifacts remain the durability anchor for this packet.
- security/compatibility doctrine: packet-filtered triage confirms strict/hardened scheduler behavior with no unclassified failures.
