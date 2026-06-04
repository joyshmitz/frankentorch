# ft-dispatch manual normalized-name builder rejection

Bead: frankentorch-ty3e
Date: 2026-06-04
Agent: codex

## Target

Profile-backed target: `schema_registry/register_1024` in `crates/ft-dispatch/benches/dispatch_bench.rs`.

After the kept digest-bucket registry index, this target remained the available non-conflicting `ft-dispatch` hotspot while other active perf beads owned `ft-kernel-cpu`, `ft-optim`, and `ft-nn`.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-dispatch --bench dispatch_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts2`

Result:

```text
schema_registry/register_1024 time: [195.00 us 195.49 us 195.99 us]
```

## Lever Tested

One lever: replace the overload arm of `OpSchemaName::unambiguous_name`:

```text
format!("{}_{}", self.base, overload)
```

with an exact-capacity `String` builder that pushes `base`, `_`, and `overload` directly.

The intended effect was to avoid `format!` overhead in the hot registration path.

## Isomorphism Proof

- Output string construction preserved exactly: `base`, underscore, overload.
- No-overload behavior stayed as `self.base.clone()`.
- Parser validation, duplicate detection, unsupported-operator ordering, registry entry order, and sorted `iter()` behavior were untouched.
- Floating point: no floating-point arithmetic was touched.
- RNG/ties: no RNG, tie-breaking, dispatch priority, or kernel selection path was touched.
- Golden outputs: existing schema-registry golden tests passed, and fixture sha256 values remained unchanged.

Golden sha256:

```text
683125fea979d9285c658d2c794eb97380d551bd14813291f14fb2b1a94d3dae  artifacts/optimization/golden_outputs/ft_dispatch_schema_registry_normalized_once_frankentorch-kgs4-15.txt
1ed775b109f682bad9b11c04e7818845bd57927ad48ef497c7ee2d6b7049207d  artifacts/optimization/golden_outputs/ft_dispatch_schema_pass21.txt
```

## Proof Commands

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-dispatch schema_registry -- --nocapture
cargo fmt -p ft-dispatch -- --check
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-dispatch --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-dispatch --all-targets --no-deps -- -D warnings
sha256sum artifacts/optimization/golden_outputs/ft_dispatch_schema_registry_normalized_once_frankentorch-kgs4-15.txt artifacts/optimization/golden_outputs/ft_dispatch_schema_pass21.txt
```

Result:

```text
cargo test: 10 passed; 0 failed; 98 filtered out
cargo fmt --check: passed
cargo check: passed
cargo clippy: passed
sha256sum: unchanged values listed above
```

Note: `ft-kernel-cpu` emitted a pre-existing duplicate `#[must_use]` warning from peer-owned work during crate-scoped builds; `ft-dispatch` passed its gates.

## Rebench

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-dispatch --bench dispatch_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts2`

Result:

```text
schema_registry/register_1024 time: [212.24 us 213.85 us 215.73 us]
```

Delta:

```text
median 195.49 us -> 213.85 us
regression 9.39%
```

Score:

```text
Impact -1 x Confidence 5 / Effort 1 = -5.0
```

Decision: reject. Source was reverted; no runtime code from this lever is kept.
