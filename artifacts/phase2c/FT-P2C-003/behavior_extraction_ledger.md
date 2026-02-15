# FT-P2C-003 — Behavior Extraction Ledger

Packet: Op schema ingestion  
Legacy anchor map: `artifacts/phase2c/FT-P2C-003/legacy_anchor_map.md`

## Behavior Families (Nominal, Edge, Adversarial)

| Behavior ID | Path class | Legacy anchor family | Strict expectation | Hardened expectation | Candidate unit/property assertions | E2E scenario seed(s) |
|---|---|---|---|---|---|---|
| `FTP2C003-B01` | nominal | schema-row parse from `native_functions.yaml` into `NativeFunction` (`torchgen/model.py:623-645`) | schema text parses deterministically into canonical function shape with stable `(name, overload)` identity | same parse contract; diagnostics may include additional context only | `ft_dispatch::schema_row_parse_round_trips_add_tensor_signature`, `ft_dispatch::operator_name_parse_preserves_overload_token` | `op_schema/strict:add_tensor_schema_roundtrip`=`16034560981891695401`, `op_schema/hardened:add_tensor_schema_roundtrip`=`7182756571925493072` |
| `FTP2C003-B02` | nominal | operator-name normalization (`BaseOperatorName`/`OperatorName`, `torchgen/model.py:2677,2793`) | inplace/dunder/overload canonicalization is deterministic and reversible | same | `ft_dispatch::base_operator_name_parse_inplace_suffix_contract`, `ft_dispatch::operator_unambiguous_name_stable` | `op_schema/strict:operator_name_normalization`=`4452319783976256027`, `op_schema/hardened:operator_name_normalization`=`12090361164728459446` |
| `FTP2C003-B03` | edge | out/structured variant metadata (`native_functions.yaml:577-593`, `Operators.h:53-60`) | out-variant schema carries explicit mutability/out-position semantics, no implicit alias repair | same; allow additional structured telemetry only | `ft_dispatch::schema_out_variant_requires_mutable_out_alias`, `ft_dispatch::structured_delegate_ref_is_preserved` | `op_schema/strict:add_out_schema_alignment`=`1443536198501121762`, `op_schema/hardened:add_out_schema_alignment`=`15896615382087176245` |
| `FTP2C003-B04` | edge | schema parser split (`parseSchemaOrName`, `function_schema_parser.cpp:408-434`) | bare-name input remains name-only; full schema requires valid argument/return grammar | same parsing behavior with bounded error categorization | `ft_dispatch::parse_schema_or_name_classifies_name_only_vs_full_schema`, `ft_dispatch::schema_parser_rejects_invalid_arrow_forms` | `op_schema/strict:parser_name_vs_schema_split`=`9239196932020020915`, `op_schema/hardened:parser_name_vs_schema_split`=`1684871375839330838` |
| `FTP2C003-B05` | adversarial | malformed/ambiguous schema text handling in parser (`function_schema_parser.cpp`) | malformed text fails closed with deterministic error code (no fallback schema inference) | same fail-closed behavior; diagnostics bounded to non-semantic context | `ft_dispatch::schema_parser_rejects_malformed_tokens`, `ft_dispatch::schema_parser_rejects_illegal_overload_name` | `op_schema/strict:malformed_schema_rejected`=`7376186281161909931`, `op_schema/hardened:malformed_schema_rejected`=`17416005490503091930` |
| `FTP2C003-B06` | adversarial | dispatch metadata mismatch (`native_functions.yaml dispatch:` vs runtime registration) | incompatible or unknown dispatch-key metadata fails closed before kernel routing | same fail-closed outcome | `ft_dispatch::schema_dispatch_keyset_rejects_unknown_backend_key`, `ft_dispatch::schema_dispatch_keyset_requires_cpu_backend_for_scoped_ops` | `op_schema/strict:dispatch_metadata_incompatible`=`10439204147922758389`, `op_schema/hardened:dispatch_metadata_incompatible`=`2697711798207276122` |
| `FTP2C003-B07` | deferred parity edge | symbolic-shape schema surface (`L-011-SYMBOLIC-SHAPE-GAP`) | unknown symbolic-shape schema behavior must not be silently accepted | same, with bounded diagnostics if surfaced | placeholder assertions gated to future packet closures (`FT-P2C-003` downstream tasks) | `op_schema/strict:symbolic_shape_gap_marker`=`11150072441506177902`, `op_schema/hardened:symbolic_shape_gap_marker`=`5960984333540654265` |

## Logging Field Expectations by Behavior Family

Mandatory deterministic replay fields (all behavior families):
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

Schema-ingestion additions:
- `op_name`
- `overload_name`
- `schema_digest`
- `dispatch_keyset_bits`
- `schema_source_ref`

Anchors:
- `crates/ft-conformance/src/logging.rs:8`
- `crates/ft-conformance/src/lib.rs:300`
- `artifacts/phase2c/UNIT_E2E_LOGGING_CROSSWALK_V1.json`

## N/A Cross-Cutting Validation Note

This ledger is docs/planning only for packet subtask A.
Execution evidence is carried by downstream packet beads:
- contract/invariant closure: `bd-3v0.14.2`
- unit/property: `bd-3v0.14.5`
- differential/adversarial: `bd-3v0.14.6`
- e2e/logging: `bd-3v0.14.7`
