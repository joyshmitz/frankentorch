# ft-serialize native decode shape-numel pass

Agent: BlackThrush
Bead: frankentorch-kgs4.18
Date: 2026-06-03

## Profile-backed target

Criterion target:

```text
native_state_dict/decode_many_small_f64_1024x4
```

Baseline via rch on `vmi1293453`:

```text
[330.47 us 336.41 us 341.42 us]
```

The decoder parsed `shape` into a `Vec<usize>`, then scanned that same shape a
second time to compute `numel`. The pass19 decode benchmark is shape-heavy
because it decodes 1024 small tensors.

## Lever

Compute `numel` while parsing shape dimensions and carry a `shape_overflow`
flag forward to the original validation point after dtype parsing.

## Behavior proof

- Tensor key ordering is unchanged: the loader still inserts into the same
  `BTreeMap` after all tensor fields are validated.
- Duplicate-key ordering is unchanged: duplicate insertion still occurs after
  values are read and tensor construction succeeds.
- Error ordering is unchanged for this lever: dimension truncation,
  dimension-to-usize conversion, and dtype validation all still run before the
  shape-overflow error is returned.
- Floating-point payload bytes are read in the same order with the same
  `from_le_bytes` conversions.
- No RNG, tie-breaking, or autograd behavior is involved.

Remote proof via rch on `vmi1293453`:

```text
cargo test -p ft-serialize native_decode_many_small_f64_golden_summary_matches_fixture -- --nocapture
1 passed

cargo test -p ft-serialize native_format_rejects_shape_overflow -- --nocapture
1 passed
```

Golden fixture sha256:

```text
93332e6e21332f43535b64ab5fe3224f22213becb95cb4d3b6e0ed9888dbe943  artifacts/optimization/golden_outputs/ft_serialize_decode_pass19.txt
```

Global golden verification:

```text
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
all present fixtures OK
```

## Benchmark delta

After via rch on `vmi1293453`:

```text
[292.90 us 300.71 us 310.05 us]
```

p50 delta: `336.41 us -> 300.71 us`, a 10.61% reduction.

Score: Impact 3 x Confidence 4 / Effort 1 = 12.0.
