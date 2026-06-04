# ft-dispatch schema registration name-digest reuse (frankentorch-ozl7)

## Target

- Skill loop: `/repeatedly-apply-skill` applying `/extreme-software-optimization`, with `/alien-graveyard` and `/alien-artifact-coding` discipline for structural pivots.
- Bead: `frankentorch-ozl7`
- Crate: `ft-dispatch`
- Benchmark: `schema_registry/register_1024`
- Profile-backed source target: `SchemaRegistry::register` hashes `normalized_name` twice for `ParsedSchemaInput::Name`.

`register_1024` builds registrations from name-only inputs such as `add.bench_0`, so the hot path enters the `ParsedSchemaInput::Name` branch. That branch computed `digest64(normalized_name)` for `schema_digest`, then the common path computed the same digest again for `name_digest` before duplicate lookup.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-dispatch --bench dispatch_bench -- schema_registry/register_1024 --warm-up-time 1 --measurement-time 10 --sample-size 20
```

Worker/result:

```text
ts1: [129.86 us 131.20 us 132.52 us]
```

## Lever

Carry `name_digest` out of the `match` in `SchemaRegistry::register`.

- For `ParsedSchemaInput::Name`, reuse the already-computed `schema_digest` as `name_digest`.
- For `ParsedSchemaInput::Schema`, preserve the previous distinction: `schema.schema_digest` remains the full schema digest, while `name_digest` is still `digest64(schema.op.unambiguous_name())`.

This is one lever: remove the redundant hot-path hash without changing registry data layout or lookup semantics.

## Isomorphism proof

- Registration ordering is unchanged: keyset validation still happens first, duplicate lookup still precedes `BinaryOp` support validation, and insertion order is untouched.
- Error ordering is unchanged: duplicate normalized names are still reported before unsupported operator bases.
- Digest semantics are unchanged:
  - Name-only inputs still use `digest64(unambiguous_name)` as both schema and name digest, exactly as before.
  - Full-schema inputs still use `schema.schema_digest` for stored schema identity and `digest64(unambiguous_name)` for name-bucket lookup.
- Tie-breaking, floating-point behavior, and RNG behavior are not involved in this path.
- Golden fixture checksums are unchanged:
  - `ft_dispatch_schema_registry_normalized_once_frankentorch-kgs4-15.txt`: `683125fea979d9285c658d2c794eb97380d551bd14813291f14fb2b1a94d3dae`
  - `ft_dispatch_schema_pass21.txt`: `1ed775b109f682bad9b11c04e7818845bd57927ad48ef497c7ee2d6b7049207d`

## Validation

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-dispatch schema_registry -- --nocapture
```

Result: pass, 10 schema-registry tests passed.

```text
cargo fmt -p ft-dispatch -- --check
```

Result: pass. `rch exec` refused the non-compilation fmt command under `RCH_REQUIRE_REMOTE=1`, so this format-only check was run locally.

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-dispatch --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-dispatch --all-targets --no-deps -- -D warnings
ubs crates/ft-dispatch/src/lib.rs
```

Result: pass. The remote check/clippy output includes an unrelated dependency warning from the active `ft-kernel-cpu` surface; `ft-dispatch` itself passed. UBS exited 0 and reported existing broad inventory warnings only.

## Re-benchmark

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-dispatch --bench dispatch_bench -- schema_registry/register_1024 --warm-up-time 1 --measurement-time 10 --sample-size 20
```

Worker/result:

```text
ts1: [125.55 us 127.88 us 130.53 us]
```

Delta:

```text
p50 131.20 us -> 127.88 us
speedup 1.026x
time reduction 2.5%
```

## Score

```text
Impact 1 x Confidence 4 / Effort 1 = 4.0
```

Kept: the change clears the `>=2.0` score threshold with same-worker RCH evidence and exact behavior proof.
