# frankentorch-6099 isomorphism proof

## Lever

`matrix_nms` now computes each selected mask row's decay factor directly while scanning higher-scoring packed masks. It no longer materializes the dense `k * k` IoU matrix.

## Behavior Preservation

- Ordering: score sort, `pre_nms_top_k` truncation, final score sort, and output index order are unchanged.
- Tie-breaking: both `partial_cmp(...).unwrap_or(Ordering::Equal)` sort comparators are unchanged.
- Floating point: for every consumed pair `(i, j)` with `j < i`, the packed-word zip order, popcount sum order, union expression, `max(1e-6)`, IoU division, `(-iou * iou / sigma).exp()`, and per-row `factor *= decay` order are unchanged.
- Parallelism: rows are independent; Rayon collection preserves row index order in the resulting `Vec<f64>`.
- RNG: `matrix_nms` reads caller-provided tensors only and has no RNG path.
- Diagnostics: shape validation and empty-input behavior happen before the changed block and are unchanged.

## Evidence

- Baseline: `matrix_nms/256x48x48` on `vmi1264463` was `[15.197 ms 19.347 ms 23.449 ms]`.
- After: `matrix_nms/256x48x48` on `vmi1227854` was `[4.0589 ms 4.2103 ms 4.4688 ms]`.
- Focused proof: `matrix_nms_parallel_match_serial_bit_exact` passed via rch on `vmi1156319`.
- Golden outputs: `golden_matrix_nms_outputs.txt` records the representative output indices and score bits; `sha256sum -c tests/artifacts/perf/20260603T1851Z-boldox-matrix-nms-stream-decay/golden_checksums.txt` passed.
- Check: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-api --all-targets` passed on `vmi1153651`.

## Gate Notes

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-api --all-targets -- -D warnings` failed before this diff on existing `ft-kernel-cpu` SVD lints: `unused_assignments` for `nm`, and `assign_op_pattern` for `rv1[i] = c * rv1[i]` / `g = c * g`.
- `cargo fmt -p ft-api --check` failed on broad pre-existing formatting drift in ft-api benches and old ft-api code; `git diff --check` passed for this diff.
- `ubs crates/ft-api/src/lib.rs` completed with the existing file-wide ft-api backlog; it did not identify a new Matrix NMS-specific finding.

## Score

Impact 5 x confidence 3 / effort 2 = 7.5. Keep.
