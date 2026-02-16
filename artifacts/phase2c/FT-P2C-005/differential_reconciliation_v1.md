# FT-P2C-005 Differential Drift Reconciliation (v1)

## Scope

- source report: `artifacts/phase2c/conformance/differential_report_v1.json`
- packet slice report: `artifacts/phase2c/FT-P2C-005/differential_packet_report_v1.json`
- packet: `FT-P2C-005`

## Result

- packet checks: `168`
- pass checks: `160`
- non-pass checks: `8`
- packet allowlisted drifts: `8`
- packet blocking drifts: `0`

Allowlisted drifts are constrained to hardened dispatch mode-split policy cases projected from FT-P2C-002:
- drift id: `dispatch.composite_backend_fallback`
- comparator: `mode_split_policy`
- cases: `autograd_without_cpu_fail_closed`, `composite_route_mode_split`, `device_mismatch_fail_closed`, `dtype_mismatch_fail_closed`, `empty_keyset_fail_closed`, `no_backend_key_fail_closed`, `no_type_key_fail_closed`, `unknown_dispatch_key_fail_closed`

No blocking drift remains in the FT-P2C-005 packet projection.

## Metamorphic + Adversarial Coverage

Metamorphic comparators executed under strict and hardened modes:
- `metamorphic_commutative_local`
- `metamorphic_offset_shift_contiguous_local`
- `metamorphic_offset_shift_contiguous_oracle`
- `metamorphic_offset_shift_linear_local`
- `metamorphic_offset_shift_linear_oracle`
- `metamorphic_offset_shift_numel_local`
- `metamorphic_offset_shift_numel_oracle`

Adversarial/fail-closed comparators executed under strict and hardened modes:
- `adversarial_autograd_without_cpu_rejected`
- `adversarial_unknown_key_rejected`
- `fail_closed`
- `fail_closed_oracle`
- `fail_closed_oracle_guard`

## Risk-Note Linkage

Validated threat controls mapped in `artifacts/phase2c/FT-P2C-005/threat_model.md`:
- `THR-501`: dispatch-key poisoning fail-closed behavior
- `THR-502`: dtype/device coercion rejection
- `THR-503`: broadcast/in-place guard enforcement
- `THR-504`: deterministic kernel output envelope
- `THR-505`: forensics envelope continuity across projected packet checks

Deferred scope remains explicit:
- `FTP2C005-B06` / `THR-506` (vectorized/quantized/sparse parity closure)

## Method-Stack Status for This Bead

- alien-artifact-coding: FT-P2C-005 packet differential projection emitted with deterministic scenario/seed lineage and explicit threat linkage.
- extreme-software-optimization: no optimization lever changed; this bead is validation-only and preserves behavior-isomorphism for all passing comparators.
- security/compatibility doctrine: strict+hardened checks remain fail-closed except bounded hardened allowlisted dispatch mode-split drift.
- RaptorQ-everywhere durability: sidecar durability updates remain delegated to final evidence-pack closure beads.
