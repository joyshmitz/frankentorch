# FT-P2C-004 Differential Drift Reconciliation (v1)

## Scope

- source report: `artifacts/phase2c/conformance/differential_report_v1.json`
- packet slice report: `artifacts/phase2c/FT-P2C-004/differential_packet_report_v1.json`
- packet: `FT-P2C-004`

## Result

- packet checks: `42`
- pass checks: `39`
- non-pass checks: `3`
- packet allowlisted drifts: `3`
- packet blocking drifts: `0`

Packet allowlisted drifts are all scoped to the hardened reentrant fallback comparator:
- `autograd.reentrant_depth_bounded_fallback` (`hardened:composite_scheduler_dac`, comparator `policy`)
- `autograd.reentrant_depth_bounded_fallback` (`hardened:dependency_underflow_probe`, comparator `policy`)
- `autograd.reentrant_depth_bounded_fallback` (`hardened:scheduler_chain`, comparator `policy`)

## Metamorphic + Adversarial Coverage

Metamorphic comparators executed under strict and hardened modes:
- `metamorphic_scale_relation_local`
- `metamorphic_scale_relation_oracle`

Adversarial fail-closed comparators executed under strict and hardened modes:
- `adversarial_strict_reentrant_overflow_rejected`
- `adversarial_hardened_reentrant_overflow_guarded`

## Risk-Note Linkage

Validated threat controls mapped in `artifacts/phase2c/FT-P2C-004/risk_note.md`:
- `THR-401`: dependency underflow corruption fails closed
- `THR-402`: scheduler ordering determinism
- `THR-403`: reentrant overflow mode-split enforcement
- `THR-404`: hardened allowlist drift containment
- `THR-405`: replay evidence continuity through differential checks

Deferred scope remains explicit:
- `autograd_scheduler:multithread_queue_interleave_gap`

## Method-Stack Status for This Bead

- alien-artifact-coding: packet differential slice emitted with deterministic mode-split drift classification and scenario linkage.
- extreme-software-optimization: no optimization lever changed; scheduler behavior-isomorphism preserved for all non-allowlisted comparators.
- security/compatibility doctrine: strict mode remains fail-closed; hardened mode drift is bounded to allowlisted `autograd.reentrant_depth_bounded_fallback`.
- RaptorQ-everywhere durability: durability sidecar updates are deferred to final packet evidence bead (`bd-3v0.15.9`).
