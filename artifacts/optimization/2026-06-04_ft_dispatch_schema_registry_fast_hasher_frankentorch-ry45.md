# ft-dispatch schema registry fast hasher - frankentorch-ry45

Date: 2026-06-04
Agent: BlackThrush
Crate: ft-dispatch
Target: `schema_registry/register_1024`

## Profile-backed target

`br ready --json` returned no ready perf beads. Active perf work owned `ft-kernel-cpu`, `ft-optim`, `ft-nn`, and `ft-api`, so this pass stayed on `ft-dispatch`.

Initial baseline:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-dispatch --bench dispatch_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
worker: vmi1149989
baseline: [207.59 us 215.83 us 226.98 us]
```

Alien-graveyard primitive: replace default `HashMap` hashing on a non-DoS internal hot path with a faster deterministic safe-Rust hasher (§7.7 high-performance hash maps; non-DoS `HashMap` replacement note).

## One lever shipped

`SchemaRegistry` now uses a private deterministic FNV-1a-style `BuildHasherDefault` for `name_to_index`.

## Isomorphism proof

This changes only bucket placement inside the private map. Schema names, duplicate detection, unsupported-operator errors, lookup keys, `SchemaDispatchEntry` contents, sorted `iter()` order, registry equality, dispatch key ordering, floating-point behavior, RNG behavior, and kernel dispatch behavior are unchanged.

Golden fixtures:

```text
683125fea979d9285c658d2c794eb97380d551bd14813291f14fb2b1a94d3dae  artifacts/optimization/golden_outputs/ft_dispatch_schema_registry_normalized_once_frankentorch-kgs4-15.txt
1ed775b109f682bad9b11c04e7818845bd57927ad48ef497c7ee2d6b7049207d  artifacts/optimization/golden_outputs/ft_dispatch_schema_pass21.txt
```

Remote proof:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-dispatch schema_registry -- --nocapture
worker: ts2
result: 9 passed; 0 failed

RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-dispatch --all-targets
worker: ts2
result: pass

cargo fmt -p ft-dispatch -- --check
result: pass

RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-dispatch --all-targets --no-deps -- -D warnings
worker: ts2
result: pass
```

Note: full dependency clippy without `--no-deps` is currently blocked by peer-owned dirty `ft-kernel-cpu` warnings unrelated to this hunk.

## Re-benchmark

Cross-worker after, used only for direction:

```text
worker: ts2
after: [290.58 us 291.64 us 292.95 us]
```

Same-worker confirmation:

```text
original on ts2: [338.80 us 339.39 us 340.05 us]
candidate on ts2: [289.45 us 290.31 us 291.00 us]
```

Median delta: 339.39 us -> 290.31 us, 1.169x faster.

Score: Impact 4 x Confidence 4 / Effort 1 = 16.0. Kept.
