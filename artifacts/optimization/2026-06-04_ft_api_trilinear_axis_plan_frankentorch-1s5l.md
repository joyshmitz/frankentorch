# ft-api trilinear axis-plan optimization

Bead: `frankentorch-1s5l`
Date: 2026-06-04
Agent: AzureTower

## Profile-backed target

Criterion target:

```text
cargo bench -p ft-api --bench ops_bench -- interpolate_trilinear/2x8x16x16x16_2x --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Baseline via rch on worker `ts2`:

```text
[4.8368 ms 5.3322 ms 5.7814 ms]
```

Hotspot: non-grad trilinear interpolation recomputed z/y/x source coordinates,
clamps, floors, and fractions inside output rows even though each axis mapping is
invariant across all batch/channel planes.

## One lever

Precompute per-output z/y/x axis plans once:

```text
LinearPlan { lo, hi, t }
```

The existing `interp_axis_coord` formula remains the single source of truth.
The fill loop only reuses precomputed low/high/fraction values and keeps the
same eight corner reads and accumulation expression order.

## Rebenchmark

Primary after run via rch on worker `ts1`:

```text
[2.0964 ms 2.1311 ms 2.1732 ms]
```

Additional directional after run on worker `vmi1152480`:

```text
[1.6181 ms 1.7327 ms 1.9025 ms]
```

The second run completed with exit 0, but custom target artifact retrieval hit
an SSH reset, so it is recorded as directional only.

Conservative p50 ratio:

```text
5.3322 ms / 2.1311 ms = 2.50x
```

Score:

```text
Impact 2.50 * Confidence 0.85 / Effort 1.0 = 2.125
```

Verdict: keep.

## Isomorphism proof

- Ordering preserved: yes. Output rows are still filled in the same row-major
  linear positions.
- Tie-breaking unchanged: N/A. No comparisons or tie rules changed.
- Floating-point preserved: yes. The same coordinate formula is called by
  `interp_axis_coord`, and the eight-corner accumulation expression keeps the
  same term order.
- RNG unchanged: yes. No RNG path touched.
- Shape and diagnostics unchanged: yes. Validation and tensor creation paths are
  before this loop and unchanged.
- Golden output: `ft_api_trilinear_axis_plan_frankentorch-1s5l.txt`
- Golden sha256:
  `110d0a6c96b7d33cc84cc23e72213ea7bca5386e73a857d5e5333f937083b0c7`

## Validation

Passed:

```text
CARGO_TARGET_DIR=/data/tmp/frankentorch-codex-1s5l-test3 rch exec -- cargo test -p ft-api interpolate_trilinear_parallel_match_serial_bit_exact -- --nocapture
CARGO_TARGET_DIR=/data/tmp/frankentorch-codex-1s5l-check rch exec -- cargo check -p ft-api --all-targets
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
git diff --check -- crates/ft-api/src/lib.rs artifacts/optimization/golden_checksums.txt artifacts/optimization/golden_outputs/ft_api_trilinear_axis_plan_frankentorch-1s5l.txt
```

Blocked by existing ft-api backlog:

```text
CARGO_TARGET_DIR=/data/tmp/frankentorch-codex-1s5l-clippy rch exec -- cargo clippy -p ft-api --all-targets -- -D warnings
CARGO_TARGET_DIR=/data/tmp/frankentorch-codex-1s5l-fmt rch exec -- cargo fmt -p ft-api --check
ubs crates/ft-api/src/lib.rs artifacts/optimization/golden_checksums.txt artifacts/optimization/golden_outputs/ft_api_trilinear_axis_plan_frankentorch-1s5l.txt
```

`cargo clippy` failed with about 200 pre-existing ft-api lint failures such as
`needless_range_loop`, `manual_is_multiple_of`, public API `too_many_arguments`,
and generated numeric constant precision warnings. `cargo fmt --check` failed
on broad pre-existing ft-api and ft-api bench formatting drift. UBS completed
after 292 seconds and reported broad existing ft-api panic/indexing/heuristic
security findings; no unsafe blocks were detected.
