# frankentorch-hwsk: ft-serialize bulk native f32 save writes

## Profile-backed target

- Bead: `frankentorch-hwsk`
- Target: `ft-serialize` native state-dict save of one contiguous F32 tensor with 1,000,000 values.
- Hotspot: the baseline writer issued one `write_all` call per F32 scalar after materializing the contiguous F32 view.
- Single lever: batch F32 values into a bounded 64 KiB scratch byte buffer and write each chunk through the existing `write_native_bytes` wrapper.

## Baseline

Command:

```bash
mkdir -p target/turquoise-pine-ft-serialize-hwsk-baseline && rch exec -- env CARGO_TARGET_DIR=target/turquoise-pine-ft-serialize-hwsk-baseline cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/save_single_f32_1m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `vmi1167313`

Criterion:

```text
native_state_dict/save_single_f32_1m time: [2.7095 ms 2.7922 ms 2.8563 ms]
```

## After

Command:

```bash
mkdir -p target/turquoise-pine-ft-serialize-hwsk-after && rch exec -- env CARGO_TARGET_DIR=target/turquoise-pine-ft-serialize-hwsk-after cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/save_single_f32_1m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `vmi1156319`

Criterion:

```text
native_state_dict/save_single_f32_1m time: [1.2766 ms 1.3617 ms 1.4775 ms]
```

Delta: mean `2.7922 ms` -> `1.3617 ms`, `2.05x` faster.

Score: Impact `3` x Confidence `2` / Effort `1` = `6.0`.

## Isomorphism proof

- Ordering: unchanged because `write_state_dict_to_writer` still iterates the caller's `BTreeMap` in key order.
- Tie-breaking: unchanged; no comparison or selection logic changed.
- Floating point: unchanged; each F32 value still uses `value.to_le_bytes()` in original slice order, preserving sign-zero, subnormal, infinity, and NaN payload bits.
- RNG: not used by the target.
- Error ordering: unchanged up to value writes; `contiguous_values_f32()` and dtype validation run before chunked writing as before. I/O errors still flow through `write_native_bytes` and `io_err`.
- Memory: the new scratch buffer is bounded at 64 KiB, avoiding a large whole-tensor byte allocation.

Golden fixture:

```text
a4f99bf82139749e11ea6a626324f0fb77a7498f797085350280c5d63fabc233  artifacts/optimization/golden_outputs/ft_serialize_f32_save_bulk_pass26.txt
```

The fixture pins the full native encoded hex plus per-key F32 bit strings for keys `a.norm` and `z.edge`.

## Verification

```bash
rch exec -- env CARGO_TARGET_DIR=target/turquoise-pine-ft-serialize-hwsk-verify cargo test -p ft-serialize native_format_f32_save_bulk_golden_summary_matches_fixture -- --nocapture
```

Result: passed on `vmi1293453` (`1 passed`).

```bash
rch exec -- env CARGO_TARGET_DIR=target/turquoise-pine-ft-serialize-hwsk-verify cargo test -p ft-serialize streaming_native_save_matches_encoder_bytes -- --nocapture
```

Result: passed on `vmi1153651` (`1 passed`).

```bash
rch exec -- env CARGO_TARGET_DIR=target/turquoise-pine-ft-serialize-hwsk-verify cargo check -p ft-serialize --all-targets
```

Result: passed on `vmi1153651`.

```bash
rch exec -- env CARGO_TARGET_DIR=target/turquoise-pine-ft-serialize-hwsk-verify cargo clippy -p ft-serialize --all-targets --no-deps -- -D warnings
```

Result: passed on `vmi1153651`.

```bash
cargo fmt -p ft-serialize --check
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
git diff --check
ubs crates/ft-serialize/src/lib.rs crates/ft-serialize/benches/serialize_bench.rs artifacts/optimization/golden_checksums.txt artifacts/optimization/golden_outputs/ft_serialize_f32_save_bulk_pass26.txt
```

Result: passed. UBS exited `0`; it reported no critical findings and only the existing ft-serialize warning inventory plus the intentional benchmark vector allocation.
