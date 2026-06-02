# Pass 21: ft-dispatch schema registry registration

- Bead: `frankentorch-6twy`
- Skill loop: `/profiling-software-performance` plus `/extreme-software-optimization`
- Crate: `ft-dispatch`
- Target benchmark: `schema_registry/register_1024`

## Profile Target

Fresh profiling after the ready perf queue was drained found a dispatcher registration hotspot: `SchemaRegistry::register` normalized each schema name, checked `BTreeMap::contains_key`, then performed a second tree lookup through `insert` on the non-duplicate path.

Baseline via rch Criterion:

```text
worker: vmi1153651
command: rch exec -- cargo bench -p ft-dispatch --bench dispatch_bench -- schema_registry/register_1024 --warm-up-time 1 --measurement-time 5 --sample-size 20
time: [879.60 us 892.71 us 907.19 us]
```

## One Lever

Replace the duplicate `contains_key` plus `insert` lookup with one `BTreeMap::entry` lookup:

- `Occupied` returns the same duplicate-schema error before operator validation
- `Vacant` preserves the same operator validation and inserted entry shape
- normalized-name ownership is taken from the vacant entry key

No other dispatcher behavior was changed.

## Isomorphism Proof

- Ordering: registry storage remains a `BTreeMap`; iteration order remains sorted by normalized schema name.
- Duplicate behavior: duplicate registration still returns `DuplicateSchema` before `BinaryOp::from_schema_base` validation.
- Unsupported-operator behavior: non-duplicate unsupported operators still fail with `UnsupportedOperator`.
- Normalized names: `OpSchemaName::unambiguous_name` is still the source of both map key and entry `normalized_name`.
- Schema digests: schema digest selection is unchanged for both name-only and full-schema input.
- Floating point: the changed path performs no floating-point arithmetic.
- RNG: the changed path uses no RNG and does not change any caller-visible RNG state.
- Tie-breaking: no ordering tie-breakers are introduced; `BTreeMap` ordering is unchanged.

Golden output:

```text
sha256: 1ed775b109f682bad9b11c04e7818845bd57927ad48ef497c7ee2d6b7049207d
file: artifacts/optimization/golden_outputs/ft_dispatch_schema_pass21.txt
```

## Result

After via rch Criterion:

```text
worker: vmi1153651
command: rch exec -- cargo bench -p ft-dispatch --bench dispatch_bench -- schema_registry/register_1024 --warm-up-time 1 --measurement-time 5 --sample-size 20
time: [661.20 us 679.86 us 706.78 us]
```

Delta:

- p50: `892.71 us -> 679.86 us`
- improvement: about 23.9 percent faster
- confidence: same-worker benchmark comparison
- score: impact 2 x confidence 3 / effort 1 = 6.0
- decision: keep

The later bench-harness cleanup removed a direct-indexing warning in setup code outside `b.iter`; the generated schema sequence and measured registration loop are unchanged.

## Gates

- `rch exec -- cargo test -p ft-dispatch schema_registry_golden_summary_matches_fixture -- --nocapture` passed
- `rch exec -- cargo bench -p ft-dispatch --bench dispatch_bench -- schema_registry/register_1024 --warm-up-time 1 --measurement-time 5 --sample-size 20` passed before and after
- `rch exec -- cargo check -p ft-dispatch --all-targets` passed
- `rch exec -- cargo clippy -p ft-dispatch --all-targets --no-deps -- -D warnings` passed
- `cargo fmt -p ft-dispatch --check` passed
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed
- `git diff --check` passed
- `ubs crates/ft-dispatch/src/lib.rs crates/ft-dispatch/benches/dispatch_bench.rs crates/ft-dispatch/Cargo.toml` passed with 0 critical findings
