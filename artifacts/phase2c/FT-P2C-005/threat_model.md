# FT-P2C-005 — Security + Compatibility Threat Model

Packet scope: CPU kernel first-wave semantics  
Subtasks: `bd-3v0.16.3` (threat model), execution linkage to `bd-3v0.16.5`/`.6`/`.7`/`.8`/`.9`

## 1) Threat Model Scope

Protected invariants:
- deterministic add/mul kernel outputs for identical scalar inputs
- deterministic dispatch selection (`selected_key`, `backend_key`, `kernel`) for fixed keysets
- fail-closed behavior for dtype/device incompatibility and invalid dispatch keysets
- strict/hardened mode split that permits bounded diagnostics only (no arithmetic repair)

Primary attack/failure surfaces:
- malformed keysets or fallback drift causing non-deterministic kernel selection
- silent dtype/device coercion that mutates observable arithmetic behavior
- broadcast/in-place compatibility bypass for destination metadata
- incomplete replay envelope fields that prevent forensic reproduction

## 2) Compatibility Envelope and Mode-Split Fail-Closed Rules

| Boundary | Strict mode | Hardened mode | Fail-closed rule |
|---|---|---|---|
| dtype/device compatibility | mismatch is terminal | mismatch is terminal | no implicit cast/repair |
| dispatch keyset decoding | invalid/unknown key bits are terminal | same | no keyset repair |
| kernel arithmetic | deterministic scalar add/mul result | identical arithmetic result | no mode-based arithmetic divergence |
| broadcast/in-place compatibility | incompatible destination metadata is terminal | same terminal outcome with bounded diagnostics | no silent reshape |
| replay/forensics envelope | required fields mandatory | required fields mandatory | missing replay fields are gate failures |

## 3) Packet-Specific Abuse Classes and Defensive Controls

| Threat ID | Abuse class | Attack surface | Impact if unmitigated | Defensive control | Strict/Hardened expectation | Unit/property fixture mapping | Failure-injection e2e scenario seed(s) |
|---|---|---|---|---|---|---|---|
| `THR-501` | dispatch-key poisoning | keyset parse + dispatch selection | wrong kernel selection and parity drift | reject malformed/unknown keysets fail-closed | strict=fail, hardened=fail | planned `dispatch_keyset_contract` + adversarial keyset checks in `ft-conformance` packet FT-P2C-005 suite | `cpu_kernel/strict:not_broadcastable_probe`=`16427831546095148586`, `cpu_kernel/hardened:not_broadcastable_probe`=`3365785779336057358` |
| `THR-502` | silent dtype/device coercion | kernel input compatibility gate | arithmetic drift and hidden precision loss | `KernelError::Incompatible` for dtype/device mismatch | strict=fail, hardened=fail | `ft-kernel-cpu` compatibility tests + packet conformance mismatch fixtures | `cpu_kernel/strict:scalar_wrapper_alpha`=`10805344276021213715`, `cpu_kernel/hardened:scalar_wrapper_alpha`=`4375420325025087270` |
| `THR-503` | broadcast/in-place guard bypass | iterator destination metadata checks | unsafe mutation semantics and output corruption | fail-closed on incompatible broadcast/in-place destination | strict=fail, hardened=fail | planned in-place shape guard assertions in packet conformance fixtures | `cpu_kernel/strict:inplace_shape_guard`=`18351655830143255747`, `cpu_kernel/hardened:inplace_shape_guard`=`16091162515159161690` |
| `THR-504` | non-deterministic kernel output | add/mul kernel execution path | replay instability + parity drift | deterministic scalar math and deterministic dispatch tuple | strict=deterministic, hardened=deterministic | `ft_kernel_cpu::add_scalar_returns_expected_value`, `ft_kernel_cpu::mul_scalar_returns_expected_value` | `cpu_kernel/strict:add_mul_scalar_core`=`14278519590404804083`, `cpu_kernel/hardened:add_mul_scalar_core`=`1329606525319445234` |
| `THR-505` | forensics envelope erosion | packet e2e structured logs | irreproducible incidents and weak audits | mandatory replay/log fields with artifact refs | strict=required, hardened=required | packet e2e field-completeness checks in `ft-conformance` | `cpu_kernel/strict:broadcast_shape_match`=`17159713884164249500`, `cpu_kernel/hardened:broadcast_shape_match`=`7684529551981593166` |
| `THR-506` | out-of-scope feature ambiguity | vectorized/quantized/sparse kernel paths | false confidence from incomplete coverage | explicit scope boundary + deferred parity plan | strict/hardened both deferred with explicit docs | deferred to later FT-P2C-005/FT-P2C-007 execution beads | `cpu_kernel/strict:vectorized_dtype_gap`=`11396097535129654157`, `cpu_kernel/hardened:vectorized_dtype_gap`=`1810572254738896262` |

## 4) Mandatory Forensic Logging + Replay Artifacts for Incidents

For packet incidents, logs must include:
- `suite_id`
- `scenario_id`
- `packet_id`
- `mode`
- `seed`
- `dispatch_key`
- `selected_kernel`
- `backend_key`
- `input_shape`
- `output_shape`
- `dtype_pair`
- `broadcast_applied`
- `artifact_refs`
- `replay_command`
- `env_fingerprint`
- `reason_code`

Required linkage chain for packet forensics:
1. packet e2e log (`artifacts/phase2c/e2e_forensics/ft-p2c-005.jsonl`)
2. packet crash triage (`artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_005_v1.json`)
3. packet failure forensics index (`artifacts/phase2c/e2e_forensics/failure_forensics_index_ft_p2c_005_v1.json`)
4. differential report (`artifacts/phase2c/conformance/differential_report_v1.json`)
5. packet differential slice (`artifacts/phase2c/FT-P2C-005/differential_packet_report_v1.json`)
6. packet reconciliation note (`artifacts/phase2c/FT-P2C-005/differential_reconciliation_v1.md`)

## 5) Residual Risks and Deferred Controls

Residual risks:
- packet scope is intentionally first-wave CPU kernels and does not yet cover the full TensorIterator dtype/vectorized/sparse surface.
- hardened diagnostics are bounded but currently narrower than full PyTorch error-context payloads.

Deferred controls and ownership:
- `bd-3v0.16.5`: unit/property coverage expansion for dtype/device, broadcast, in-place adversarial probes.
- `bd-3v0.16.6`: differential/metamorphic/adversarial reconciliation and allowlist-policy checks.
- `bd-3v0.16.7`: packet e2e replay/forensics artifact emission (`ft-p2c-005` slice).
- `bd-3v0.16.8`: packet performance optimization with behavior-isomorphism evidence.
- `bd-3v0.16.9`: final evidence-pack closure with parity+RaptorQ artifacts.

## 6) N/A Cross-Cutting Validation Note

This bead is docs/planning for packet subtask C (`bd-3v0.16.3`).  
Execution evidence ownership is explicitly delegated to:
- unit/property: `bd-3v0.16.5`
- differential/adversarial: `bd-3v0.16.6`
- e2e/replay forensics: `bd-3v0.16.7`
- optimization/isomorphism: `bd-3v0.16.8`
- final evidence pack: `bd-3v0.16.9`
