# frankentorch-igz7 rejection evidence

## Target

`ft-api` Matrix NMS on `ops_bench/matrix_nms/256x48x48`.

Baseline command:

```text
RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/tmp/frankentorch-boldox-matrix-nms-baseline-target rch exec -- cargo bench -p ft-api --bench ops_bench -- matrix_nms/256x48x48 --warm-up-time 1 --measurement-time 5 --sample-size 10
```

Baseline worker `vmi1153651`:

```text
matrix_nms/256x48x48 time [8.4339 ms 9.3166 ms 10.321 ms]
```

## Attempted Lever

Compute and store only the lower-triangle Matrix NMS IoU entries (`j < i`) because the unchanged decay loop consumes only those higher-scoring mask entries.

## Isomorphism Proof

- Ordering preserved: score sort, pre-top-k truncation, decay row order, final score sort, and output index order were unchanged.
- Tie-breaking preserved: all `sort_by(...partial_cmp...)` comparators were unchanged.
- Floating point preserved for used pairs: each consumed IoU used the same packed-word popcount order, same `union_count` expression, same `max(1e-6)`, and same `(-iou * iou / sigma).exp()` order.
- RNG preserved: no RNG path exists inside `matrix_nms`; only caller-provided tensors are read.
- Golden output: the representative Matrix NMS output fixture remains the prior packed-mask fixture sha256 `ae9fe635cef260ef124ec891e5660d841bb47ce8d618b5c0a23cabc0d0d02bec`; no kept golden change was committed because the lever was rejected and reverted.
- Focused remote proof: `RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/tmp/frankentorch-boldox-matrix-nms-test-target rch exec -- cargo test -p ft-api matrix_nms_parallel_match_serial_bit_exact -- --nocapture` passed on `vmi1149989`.

## Re-benchmark

After command:

```text
RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/tmp/frankentorch-boldox-matrix-nms-after-target rch exec -- cargo bench -p ft-api --bench ops_bench -- matrix_nms/256x48x48 --warm-up-time 1 --measurement-time 5 --sample-size 10
```

After worker `vmi1167313`:

```text
matrix_nms/256x48x48 time [7.9518 ms 8.6150 ms 9.4445 ms]
```

The cross-worker p50 delta was only `9.3166 ms -> 8.6150 ms` (`1.08x`). This does not clear the Score>=2.0 keep gate under shared-worker variance.

Score: impact 0.5 x confidence 2 / effort 1 = 1.0.

## Decision

Rejected. The source lever was manually reverted before commit.

Next Matrix NMS attack should replace the representation or algorithmic flow, for example streaming decay accumulation that eliminates the full dense IoU matrix allocation, or compressed/sparse mask runs for masks with low foreground density. Do not continue with another bookkeeping-only triangle tweak.
