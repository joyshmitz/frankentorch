# FT-P2C-005 — Behavior Extraction Ledger

Packet: CPU kernel first-wave semantics  
Legacy anchor map: `artifacts/phase2c/FT-P2C-005/legacy_anchor_map.md`

## Behavior Families (Nominal, Edge, Adversarial)

| Behavior ID | Path class | Legacy anchor family | Strict expectation | Hardened expectation | Candidate unit/property assertions | E2E scenario seed(s) |
|---|---|---|---|---|---|---|
| `FTP2C005-B01` | nominal | `TensorIterator::binary_op` + CPU `binary_op_scalar` | add/mul numeric outputs match oracle for scoped scalar cases | same arithmetic outputs | `ft_kernel_cpu::add_scalar_returns_expected_value`, `ft_kernel_cpu::mul_scalar_returns_expected_value` | `cpu_kernel/strict:add_mul_scalar_core`=`14278519590404804083`, `cpu_kernel/hardened:add_mul_scalar_core`=`1329606525319445234` |
| `FTP2C005-B02` | nominal | `compute_shape` + `compute_strides` | broadcast-compatible shapes map to deterministic output shape/stride contract | same, with bounded diagnostics metadata only | planned conformance shape assertions in `ft-conformance` packet suite | `cpu_kernel/strict:broadcast_shape_match`=`17159713884164249500`, `cpu_kernel/hardened:broadcast_shape_match`=`7684529551981593166` |
| `FTP2C005-B03` | edge | add/mul scalar wrappers (`add(..., alpha)`, `mul(..., scalar)`) | dtype/device compatibility checked before kernel execution; no silent coercion drift | bounded diagnostics allowed, no arithmetic drift | planned fixture rows for scalar-wrapper parity with explicit expected kernel key | `cpu_kernel/strict:scalar_wrapper_alpha`=`10805344276021213715`, `cpu_kernel/hardened:scalar_wrapper_alpha`=`4375420325025087270` |
| `FTP2C005-B04` | edge (in-place/out variants) | iterator setup at `binary_op(self,self,other)` and out-of-place paths | in-place path rejects incompatible broadcast targets fail-closed | same fail-closed behavior with explicit reason code | planned adversarial tests for in-place shape mismatch and alias guard | `cpu_kernel/strict:inplace_shape_guard`=`18351655830143255747`, `cpu_kernel/hardened:inplace_shape_guard`=`16091162515159161690` |
| `FTP2C005-B05` | adversarial | invalid-broadcast path (`test_not_broadcastable`) | non-broadcastable inputs fail deterministically | same failure class with hardened diagnostics envelope | planned conformance adversarial comparator `invalid_broadcast_fail_closed` | `cpu_kernel/strict:not_broadcastable_probe`=`16427831546095148586`, `cpu_kernel/hardened:not_broadcastable_probe`=`3365785779336057358` |
| `FTP2C005-B06` | deferred parity edge | full TensorIterator dtype/vectorized matrix + quantized kernels | out of first-wave scope; unsupported matrix entries must be explicit | same | deferred to later FT-P2C-005 subtasks and follow-on packet chain | `cpu_kernel/strict:vectorized_dtype_gap`=`11396097535129654157`, `cpu_kernel/hardened:vectorized_dtype_gap`=`1810572254738896262` |

## Logging Field Expectations by Behavior Family

Mandatory deterministic replay fields (all CPU-kernel behavior families):
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

Kernel-specific additions:
- `dispatch_key`
- `selected_kernel`
- `backend_key`
- `input_shape`
- `output_shape`
- `dtype_pair`
- `broadcast_applied`
- `fallback_path`

Anchors:
- `crates/ft-kernel-cpu/src/lib.rs`
- `crates/ft-dispatch/src/lib.rs`
- `crates/ft-conformance/src/lib.rs`
- `artifacts/phase2c/UNIT_E2E_LOGGING_CROSSWALK_V1.json`

## N/A Cross-Cutting Validation Note

This ledger is docs/planning only for packet subtask A (`bd-3v0.16.1`).
Execution evidence ownership is carried by downstream packet beads:
- unit/property: `bd-3v0.16.5`
- differential/adversarial: `bd-3v0.16.6`
- e2e/logging: `bd-3v0.16.7`
