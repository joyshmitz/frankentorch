# FT-P2C-005 — Contract Table + Strict/Hardened Invariant Spec

Packet scope: CPU kernel first-wave semantics (scoped elementwise add/mul on CPU)

## 1) Kernel Interface Contracts (Scoped)

| Contract ID | Input | Output | Error Contract | Deterministic Invariant |
|---|---|---|---|---|
| `CPU-KERNEL-001` | `ScalarTensor lhs`, `ScalarTensor rhs` | scalar add result tensor | `KernelError::Incompatible` on dtype/device mismatch | result value is deterministic for identical inputs |
| `CPU-KERNEL-002` | `ScalarTensor lhs`, `ScalarTensor rhs` | scalar mul result tensor | `KernelError::Incompatible` on dtype/device mismatch | result value is deterministic for identical inputs |
| `CPU-KERNEL-003` | dispatch keyset + op (`add`/`mul`) | `DispatchDecision` + result tensor | `DispatchError::Key` for invalid keysets; `DispatchError::Kernel` for kernel incompatibility | selected key + backend key + kernel string remain replay-stable |
| `CPU-KERNEL-004` | iterator shape/stride config (legacy oracle anchor) | broadcasted traversal plan | non-broadcastable pairs are terminal (fail-closed) | shape/stride derivation is deterministic for fixed input metadata |
| `CPU-KERNEL-005` | in-place/out-variant binary op request | mutated/output tensor | incompatible destination shape or metadata must fail-closed | no silent reshape/repair for in-place destination |
| `CPU-KERNEL-006` | packet-level conformance fixture case | case report + forensic log | any output/dispatch mismatch is recorded as failure | forensic envelope fields remain complete and reproducible |

## 2) Strict/Hardened Invariant Spec

| Invariant ID | Strict Mode | Hardened Mode | Allowlist / Policy Link |
|---|---|---|---|
| `INV-KERNEL-MATH` | add/mul scalar arithmetic must match oracle expectations | same arithmetic outputs | no divergence permitted |
| `INV-DTYPE-DEVICE-COMPAT` | dtype/device mismatch fails-closed (`KernelError::Incompatible`) | same fail-closed behavior | no divergence permitted |
| `INV-DISPATCH-RESOLUTION` | keyset must resolve to deterministic backend/type keys | same deterministic resolution | no divergence permitted |
| `INV-BROADCAST-VALIDITY` | non-broadcastable shapes are terminal failures | same failure semantics; diagnostics may be enriched | no behavior-altering repair permitted |
| `INV-INPLACE-GUARD` | in-place destination shape/metadata must be compatible | same | no divergence permitted |
| `INV-FORENSIC-COMPLETENESS` | all replay fields required in packet logs | same | missing fields are reliability-gate failures |
| `INV-HARDENED-BOUNDARY` | no implicit fallback path for kernel arithmetic | bounded diagnostics only; arithmetic/dispatch outcomes must remain identical | any hardened drift requires explicit allowlist entry |

## 3) Error Contract Matrix

| Error Type | Trigger | Strict Behavior | Hardened Behavior |
|---|---|---|---|
| `KernelError::Incompatible(TensorCompatError::DTypeMismatch)` | lhs/rhs dtype mismatch | fail-closed | fail-closed |
| `KernelError::Incompatible(TensorCompatError::DeviceMismatch)` | lhs/rhs device mismatch | fail-closed | fail-closed |
| `DispatchKeyError::{EmptySet,NoTypeKey,NoBackendKey,UnknownBits,IncompatibleSet}` | invalid dispatch keyset | fail-closed | fail-closed |
| non-broadcastable shape (legacy iterator contract) | incompatible shape pair for binary op | fail-closed | fail-closed with bounded diagnostics envelope |

## 4) Deterministic Dispatch/Forensics Contract

`DispatchDecision` and packet-level forensic logs must expose replay-stable fields:
- `op`
- `mode`
- `selected_key`
- `backend_key`
- `kernel`
- `keyset_bits`
- `fallback_used`
- `suite_id`
- `scenario_id`
- `packet_id`
- `seed`
- `artifact_refs`
- `replay_command`

Determinism requirements:
1. identical inputs + identical dispatch keyset => identical dispatch decision fields.
2. strict/hardened modes must not diverge in kernel math outputs for scoped first-wave operations.
3. non-broadcastable and incompatible metadata paths are fail-closed in both modes.
