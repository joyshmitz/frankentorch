# FT-P2C-003 — Legacy Anchor Map

Packet: Op schema ingestion  
Legacy root: `legacy_pytorch_code/pytorch`

## Extracted Anchors (Exact)

| Legacy path | Line anchor | Symbol / region | Porting relevance | Confidence |
|---|---:|---|---|---|
| `torchgen/model.py` | 100 | `class DispatchKey(Enum)` | canonical codegen dispatch-key namespace used when binding schema rows to backend registrations | high |
| `torchgen/model.py` | 506 | `class NativeFunction` | normalized in-memory representation of each `native_functions.yaml` row | high |
| `torchgen/model.py` | 623-645 | `NativeFunction.from_yaml(...)` + `FunctionSchema.parse(...)` call | exact ingestion path from YAML text into typed schema model | high |
| `torchgen/model.py` | 1506 | `class FunctionSchema` | canonical structured schema contract (`name`, `arguments`, `returns`) used by codegen and validation | high |
| `torchgen/model.py` | 1536-1548 | `FunctionSchema.parse` | parse + round-trip assertion (`str(r) == func`) for deterministic schema normalization | high |
| `torchgen/model.py` | 2677 | `class BaseOperatorName` | overload/inplace/dunder normalization for operator identity | high |
| `torchgen/model.py` | 2793 | `class OperatorName` | final `(base, overload)` identity and unambiguous naming | high |
| `aten/src/ATen/native/native_functions.yaml` | 554 | `- func: add.Tensor(...)` | concrete schema row shape with overload token and dispatch metadata | high |
| `aten/src/ATen/native/native_functions.yaml` | 577 | `- func: add.out(...)` (`structured: True`) | out-variant schema and structured-kernel metadata boundary | high |
| `aten/src/ATen/native/native_functions.yaml` | 622 | `- func: add.Scalar(...)` | scalar overload family and explicit `CompositeExplicitAutograd` dispatch path | medium |
| `aten/src/ATen/core/operator_name.h` | 16 | `struct OperatorName` | runtime operator identity (`name`, `overload_name`) used by dispatcher registration and schema comparison | high |
| `aten/src/ATen/core/function_schema.h` | 229 | `struct FunctionSchema` | runtime schema object with compatibility checks and mutability/alias semantics | high |
| `torch/csrc/jit/frontend/function_schema_parser.h` | 15 | `parseSchemaOrName(...)` declaration | frontend entrypoint for parsing schema string vs bare op name | high |
| `torch/csrc/jit/frontend/function_schema_parser.cpp` | 408 | `parseSchemaOrName(...)` implementation | authoritative parser boundary used by registration stack | high |
| `aten/src/ATen/core/op_registration/op_registration.h` | 109 | `Options::schema(...)` | operator registration path that calls parser and stores parsed schema/name | high |
| `aten/src/ATen/templates/Operators.h` | 53-60 | `ATEN_OP*` metadata comments (`schema`, `schema_str`) | generated API contract exposing schema metadata to call sites | medium |

## Implemented Mapping (Current + Planned)

| Rust crate | File | Mapping |
|---|---|---|
| `ft-dispatch` | `crates/ft-dispatch/src/lib.rs` | current proto operator surface (`BinaryOp`) + deterministic dispatch-key routing; planned destination for typed op-schema ingestion map |
| `ft-kernel-cpu` | `crates/ft-kernel-cpu/src/lib.rs` | concrete kernel endpoints (`add_scalar`, `mul_scalar`) currently bound manually, later to be schema-indexed |
| `ft-conformance` | `crates/ft-conformance/fixtures/dispatch_key_cases.json` | existing dispatch contract fixtures that will be extended with op-schema ingestion fixtures for FT-P2C-003 |

Behavior extraction ledger:
- `artifacts/phase2c/FT-P2C-003/behavior_extraction_ledger.md`

## Confidence Notes and Undefined Regions

- `high` confidence anchors map directly to parser/model code used by PyTorch operator schema ingestion and runtime registration.
- `medium` confidence anchors involve generated-template surfaces or limited-scope overload rows used here as representative anchors.
- Undefined/deferred zones for this packet:
  - full symbolic-shape schema parity remains deferred under ledger row `L-011-SYMBOLIC-SHAPE-GAP`.
  - non-CPU backend schema expansion remains deferred to backend-focused packet chain.

## Extraction Schema (Mandatory)

1. `packet_id`: `FT-P2C-003`
2. `legacy_paths`: `torchgen/model.py`, `aten/src/ATen/native/native_functions.yaml`, `aten/src/ATen/core/{operator_name.h,function_schema.h}`, `torch/csrc/jit/frontend/function_schema_parser.cpp`
3. `legacy_symbols`: `DispatchKey`, `NativeFunction`, `FunctionSchema`, `OperatorName`, `parseSchemaOrName`
4. `op_schema_contract`: schema strings must round-trip and preserve overload + mutability semantics
5. `dispatch_contract`: schema dispatch metadata must map deterministically to backend kernel binding decisions
6. `error_contract`: invalid schema text or incompatible overload metadata fail closed
7. `strict_mode_policy`: no permissive schema repair or inferred aliasing outside explicit schema fields
8. `hardened_mode_policy`: bounded diagnostics allowed, but no behavior-altering schema repair
9. `excluded_scope`: full operator universe ingestion, non-CPU backend families, symbolic-shape parity closure
10. `oracle_tests`: `test/test_ops.py`, `test/test_dispatch.py` (schema and dispatch integration behavior families)
11. `performance_sentinels`: schema parse throughput, schema cache hit ratio, dispatch lookup latency by overload
12. `compatibility_risks`: overload-name drift, schema parser ambiguity, alias/mutability annotation mismatch
13. `raptorq_artifacts`: packet parity/evidence outputs must emit sidecar + decode proof in final evidence bead (`bd-3v0.14.9`)
