# ft-serialize Native F32 Save Bulk Writes

- Bead: `frankentorch-hwsk`
- Agent: `TurquoisePine`
- Skill loop: `/repeatedly-apply-skill` applying `/extreme-software-optimization`
- Target: native state-dict save for one contiguous F32 tensor with 1,000,000 values
- Lever: batch F32 value-byte writes into 64 KiB chunks

## Profile-Backed Target

After the ready queue drained and active kernel/optimizer perf beads were claimed by other agents, the fallback profile target was native state-dict save. The old F32 path wrote each scalar through `write_native_bytes`, so a large contiguous F32 tensor made 1,000,000 tiny `write_all` calls even with a buffered writer.

Baseline Criterion artifact:

```text
target/turquoise-pine-ft-serialize-hwsk-baseline/criterion/native_state_dict/save_single_f32_1m/new/estimates.json
```

Baseline command:

```text
rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/save_single_f32_1m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Baseline result:

```text
mean:   2.809513 ms
CI:     [2.714829 ms 2.905417 ms]
median: 2.813118 ms
```

## Change

The F32 branch still obtains the same contiguous F32 slice after the same validation. It now sends that slice to `write_native_f32_values`, which converts values to little-endian bytes in order into a reusable 64 KiB scratch buffer, then calls the same `write_native_bytes` helper once per chunk.

The benchmark adds `native_state_dict/save_single_f32_1m`.

## Isomorphism Proof

- Ordering: unchanged; `BTreeMap` key order, tensor order, shape order, and per-value order are preserved.
- Tie-breaking: N/A; no comparisons or selection logic changed.
- Floating point: unchanged; values are never recomputed, and each scalar still uses `f32::to_le_bytes`.
- RNG: N/A; no random draw path is involved.
- Error classes: unchanged; validation, unsupported dtype handling, and I/O errors still flow through the existing `TensorIOError` paths.
- Byte format: unchanged; FTSV magic/version, key bytes, ndim/shape bytes, dtype tags, and little-endian value bytes are identical.

Golden fixture:

```text
artifacts/optimization/golden_outputs/ft_serialize_f32_save_bulk_pass26.txt
```

Golden sha256:

```text
a4f99bf82139749e11ea6a626324f0fb77a7498f797085350280c5d63fabc233
```

Proof commands:

```text
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
rch exec -- cargo test -p ft-serialize native_format_f32_save_bulk_golden_summary_matches_fixture -- --nocapture
```

Results:

```text
sha256sum: passed for the F32 save fixture
ft-serialize golden test: 1 passed
```

## Bench Delta

Confirmation re-bench command:

```text
CARGO_TARGET_DIR=target/turquoise-pine-ft-serialize-hwsk-after-confirm rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/save_single_f32_1m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Confirmation after result:

```text
worker: vmi1293453
time:   [717.19 us 727.73 us 740.41 us]
mean:   0.723594 ms
CI:     [0.712035 ms 0.735668 ms]
```

Integrated delta:

```text
mean: 2.809513 ms -> 0.723594 ms
elapsed: 74.2% faster
throughput: 3.88x
```

Score:

```text
Impact 4 * Confidence 4 / Effort 1 = 16.0
```

Verdict: keep.

## Gates

Passed:

```text
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
git diff --check -- crates/ft-serialize/src/lib.rs crates/ft-serialize/benches/serialize_bench.rs artifacts/optimization/golden_checksums.txt artifacts/optimization/golden_outputs/ft_serialize_f32_save_bulk_pass26.txt
rch exec -- cargo test -p ft-serialize native_format_f32_save_bulk_golden_summary_matches_fixture -- --nocapture
rch exec -- cargo fmt -p ft-serialize --check
rch exec -- cargo check -p ft-serialize --all-targets
rch exec -- cargo clippy -p ft-serialize --all-targets --no-deps -- -D warnings
```

UBS:

```text
ubs crates/ft-serialize/src/lib.rs crates/ft-serialize/benches/serialize_bench.rs artifacts/optimization/2026-06-03_ft_serialize_f32_save_bulk_frankentorch-hwsk.md artifacts/optimization/golden_outputs/ft_serialize_f32_save_bulk_pass26.txt artifacts/optimization/golden_checksums.txt .skill-loop-progress-TurquoisePine.md
```

UBS exited 0. It reported existing warning/info inventory in `ft-serialize`, but no critical findings, and its built-in formatting, clippy, cargo check, test-build, cargo-audit, and cargo-deny probes passed.
