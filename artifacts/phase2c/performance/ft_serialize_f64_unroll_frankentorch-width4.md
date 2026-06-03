# ft-serialize width-4 f64 decode pass

- Agent: TurquoisePine
- Crate: `ft-serialize`
- Target benchmark: `native_state_dict/decode_many_small_f64_1024x4`
- Skill loop: `/repeatedly-apply-skill` applying `/extreme-software-optimization`
- Candidate source path: `load_state_dict_from_bytes` -> `read_f64_payload`

## Profile target

The native FTSV state-dict benchmark decodes 1024 tensors with shape `[4]` and
dtype F64. Earlier decode passes removed the duplicate `BTreeMap` lookup and
shape/numel rescan; the residual profile target remains the fixed-width F64
payload reader for many tiny width-4 tensors.

Baseline via rch Criterion:

```text
worker: vmi1153651
command: rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 5 --sample-size 20
time: [705.03 us 732.03 us 763.07 us]
outliers: 1 low mild, 1 high severe
```

## One lever

After `native_payload` has performed byte-count overflow checks, truncation
checks, and advanced the cursor, add a `numel == 4` F64 fast path that constructs
the four values directly from fixed little-endian byte offsets. The general
`chunks_exact(8)` loop remains unchanged for all other tensor widths.

## Isomorphism proof

- Ordering: key parsing, duplicate-key rejection, shape parsing, dtype dispatch,
  `BTreeMap` insertion, and trailing-byte checks remain outside this helper.
- Error order: byte-count overflow and truncation are still handled by
  `native_payload` before the fast path runs.
- Floating point: every value still uses `f64::from_le_bytes` on the same byte
  lanes; no arithmetic, cast, comparison, or canonicalization is introduced.
- Raw bits: focused tests decode signed zero, subnormal, infinities, quiet NaN
  payload, and signaling-NaN-like payload through width-4 tensors and compare
  `to_bits()` exactly.
- RNG and tie-breaking: the decoder uses no RNG and performs no ordering
  comparisons inside the payload reader.

Golden output verification:

```text
93332e6e21332f43535b64ab5fe3224f22213becb95cb4d3b6e0ed9888dbe943  artifacts/optimization/golden_outputs/ft_serialize_decode_pass19.txt
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing: OK for present fixtures
```

## Result

After via rch Criterion:

```text
worker: vmi1149989
command: rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 5 --sample-size 20
time: [308.27 us 355.84 us 429.97 us]
```

Delta:

- p50: `732.03 us -> 355.84 us`
- improvement: about 51.4 percent faster
- confidence: capped for cross-worker comparison and noisy baseline, but the
  measured delta is above the keep threshold
- score: Impact 3 x Confidence 2 / Effort 1 = 6.0
- decision: keep

## Gates

- `rch exec -- cargo test -p ft-serialize native_format_ -- --nocapture` passed
  10 native-format tests, including width-4 raw-bit and truncation regressions.
- `rch exec -- cargo check -p ft-serialize --all-targets` passed.
- `rch exec -- cargo clippy -p ft-serialize --all-targets --no-deps -- -D warnings`
  passed.
- `rch exec -- cargo fmt -p ft-serialize --check` passed; rch warned that this
  is a non-compilation command.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`
  passed for present fixtures.
- `sha256sum -c artifacts/phase2c/performance/ft_serialize_f64_unroll_frankentorch-width4.sha256`
  passed.
- `git diff --check` passed.
- `ubs crates/ft-serialize/src/lib.rs crates/ft-serialize/benches/serialize_bench.rs crates/ft-serialize/Cargo.toml`
  passed with exit 0 and no critical issues. UBS reported 522 warnings and 179
  info items from existing inventory plus test-only panic/allocation patterns;
  its own fmt, clippy, cargo check, tests-build, cargo-audit, and cargo-deny
  sections were clean.
