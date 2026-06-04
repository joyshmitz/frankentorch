# ft-core TensorMeta numel inline rejection - frankentorch-t9vt

Date: 2026-06-04
Agent: BlackThrush
Crate: ft-core
Target: `tensor_meta/numel_rank8_repeated_65536`

## Profile-backed target

`br ready --json` returned no ready perf beads, and current in-progress perf work was owned on other crates. I used the existing ft-core Criterion hotspot:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-core --bench core_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
worker: ts2
baseline: [28.967 us 28.998 us 29.044 us]
```

## One lever tested

Added `#[inline(always)]` to `TensorMeta::numel`.

## Isomorphism proof

Behavior surface is unchanged by construction: the candidate only changed an inlining hint on a getter that returns the already cached `self.numel`; it did not alter shape ordering, stride ordering, tie-breaking, floating-point operations, RNG, allocation layout, or overflow semantics.

Golden fixture:

```text
sha256sum artifacts/optimization/golden_outputs/ft_core_numel_pass22.txt
fc134f8a2fbb18b29efcf4f4d8c09d3e78c8a0b4375fc586fb9035a658bff864
```

Remote proof:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-core numel -- --nocapture
worker: ts2
result: 4 passed; 0 failed

RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-core --all-targets
result: pass

cargo fmt -p ft-core -- --check
result: pass

RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-core --all-targets -- -D warnings
worker: ts2
result: pass
```

## Re-benchmark

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-core --bench core_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
worker: ts2
after: [28.978 us 29.027 us 29.081 us]
```

Median delta: 28.998 us -> 29.027 us, a 0.10% regression.

Score: Impact 0 x Confidence 5 / Effort 1 = 0.0. Rejected; source hunk reverted.
