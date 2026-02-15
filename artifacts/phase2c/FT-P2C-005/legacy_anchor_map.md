# FT-P2C-005 — Legacy Anchor Map

Packet: CPU kernel first-wave semantics  
Legacy root: `legacy_pytorch_code/pytorch`

## Extracted Anchors (Exact)

| Legacy path | Line anchor | Symbol | Porting relevance |
|---|---:|---|---|
| `aten/src/ATen/native/BinaryOps.cpp` | 813 | `TensorIterator::binary_op(result, self, other)` | canonical binary-op iterator construction for tensor-tensor add/mul paths |
| `aten/src/ATen/native/BinaryOps.cpp` | 992 | `Tensor mul(const Tensor& self, const Scalar& other)` | scalar-wrapper redispatch path for mul |
| `aten/src/ATen/native/BinaryOps.cpp` | 1176 | `Tensor add(const Tensor& self, const Scalar& other, const Scalar& alpha)` | scalar-wrapper + alpha contract for add |
| `aten/src/ATen/native/BinaryOps.cpp` | 1308 | out-of-place add iterator setup | output shape/stride compatibility envelope |
| `aten/src/ATen/native/BinaryOps.cpp` | 1322 | in-place add iterator setup (`binary_op(self, self, other)`) | in-place aliasing boundary |
| `aten/src/ATen/native/BinaryOps.cpp` | 1356 | out-of-place mul iterator setup | output buffer semantics for mul |
| `aten/src/ATen/native/BinaryOps.cpp` | 1370 | in-place mul iterator setup (`binary_op(self, self, other)`) | in-place mutation/shape guard contract |
| `aten/src/ATen/native/BinaryOps.cpp` | 378 | `DEFINE_DISPATCH(mul_stub)` | dispatcher-to-kernel binding for mul |
| `aten/src/ATen/native/BinaryOps.cpp` | 437 | `add_stub(device_type(), *this, -alpha)` | add backend dispatch with alpha adjustment |
| `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp` | 31 | `binary_op_scalar` helper | scalar fast-path kernel pattern |
| `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp` | 61 | `vec::fmadd(...)` in add kernel path | vectorized fused-add shape of CPU kernel implementation |
| `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp` | 1448 | `REGISTER_DISPATCH(mul_stub, &mul_kernel)` | CPU mul kernel registration endpoint |
| `aten/src/ATen/TensorIterator.cpp` | 1237 | `TensorIteratorBase::compute_shape` | broadcasted output-shape derivation |
| `aten/src/ATen/TensorIterator.cpp` | 1277 | `TensorIteratorBase::compute_strides` | per-operand stride/materialization contract |
| `aten/src/ATen/TensorIterator.cpp` | 1493 | `TensorIteratorBase::build` | iterator config pipeline before kernel execution |
| `aten/src/ATen/TensorIterator.cpp` | 763 | `TensorIteratorBase::for_each` | execution traversal and grain-size split behavior |

## Behavioral Oracle Tests (Scoped)

| Legacy path | Intent |
|---|---|
| `test/test_binary_ufuncs.py` (`test_add`, `test_mul`, `test_broadcasting`, `test_not_broadcastable`) | binary op math, alpha behavior, broadcasting, and invalid-broadcast failures |
| `test/test_torch.py` (`test_broadcast`) | generic broadcast compatibility assertions used across binary operators |
| `test/test_binary_ufuncs.py` (`test_add_broadcast_empty`) | empty-shape/broadcast edge behavior |

## Implemented Rust Mapping

| Rust crate | File | Mapping |
|---|---|---|
| `ft-kernel-cpu` | `crates/ft-kernel-cpu/src/lib.rs` | first-wave scalar CPU kernels (`add_scalar`, `mul_scalar`) and deterministic error conversion |
| `ft-dispatch` | `crates/ft-dispatch/src/lib.rs` | dispatch decision envelope for backend key and kernel identity |
| `ft-core` | `crates/ft-core/src/lib.rs` | dtype/device/tensor metadata contract consumed by kernels |
| `ft-conformance` | `crates/ft-conformance/src/lib.rs` | packet-scoped conformance/differential/e2e hooks (to be expanded for FT-P2C-005)|

## Extraction Schema (Mandatory)

1. `packet_id`: `FT-P2C-005`
2. `legacy_paths`: `aten/src/ATen/native/BinaryOps.cpp`, `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp`, `aten/src/ATen/TensorIterator.cpp`
3. `legacy_symbols`: `TensorIterator::binary_op`, `compute_shape`, `compute_strides`, `add_stub`, `mul_stub`, `binary_op_scalar`
4. `tensor_storage_contract`: elementwise binary ops preserve deterministic scalar math and explicit dtype/device checks
5. `dispatch_contract`: CPU backend dispatch must resolve to deterministic kernel identity (no implicit fallback drift)
6. `error_contract`: non-broadcastable/invalid-shape paths fail closed (no silent shape repair)
7. `grad_graph_contract`: packet scope is kernel semantics; autograd linkage remains external but must preserve kernel-observable outputs
8. `serialization_contract`: no packet-local serialization format change
9. `strict_mode_policy`: strict compatibility with fail-closed invalid shape/dtype behavior
10. `hardened_mode_policy`: bounded diagnostics permitted, no behavior-altering repair for kernel arithmetic
11. `excluded_scope`: quantized kernels, sparse kernels, full vectorized dtype matrix, CUDA/MPS kernel paths
12. `oracle_tests`: `test/test_binary_ufuncs.py`, `test/test_torch.py` broadcast and binary-op subsets
13. `performance_sentinels`: per-op p50/p95/p99 latency, throughput under representative elementwise traces
14. `compatibility_risks`: first-wave scalar kernels currently narrower than full PyTorch TensorIterator dtype/broadcast surface
15. `raptorq_artifacts`: packet parity sidecar + decode proof remain mandatory in final evidence bead (`bd-3v0.16.9`)
