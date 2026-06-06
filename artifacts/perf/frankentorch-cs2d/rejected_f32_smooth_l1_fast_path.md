# frankentorch-cs2d: f32 smooth-L1 fast path probe rejected/not kept

## Target

`tensor_smooth_l1_loss` has a fused same-shape f64 path, while same-shape f32
no-grad routed through the generic op graph.

## Baseline evidence

Command:

```bash
rch exec -- cargo bench -p ft-api --bench ops_bench -- smooth_l1/nograd_f32_8m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts1`

Result: the pre-change f32 path did not produce a Criterion baseline. It failed
during warmup with:

```text
tensor comparison requires matching f32 or f64 dtypes
```

This happens because the f32 fallback constructs f64 `full()` constants and then
calls the comparison path with mixed f32/f64 operands.

Reference f64 no-grad benchmark:

```text
smooth_l1/nograd_8m time: [60.428 ms 62.225 ms 63.841 ms]
```

Worker: `vmi1227854`

## Candidate

Single lever attempted locally and in a scratch copy:

- add `smooth_l1_forward_f32` in `ft-kernel-cpu`
- route same-shape f32 no-grad `tensor_smooth_l1_loss` through it
- keep reductions delegated to `tensor_mean`/`tensor_sum`
- preserve the `abs(d) < beta` branch rule and f32 dtype

## Proof status

Scratch RCH proof on `vmi1167313`:

```text
test tests::smooth_l1_forward_f32_matches_piecewise_formula ... ok
1 passed
```

The earlier `api_f32_smooth_l1_test*.txt` and
`smooth_l1_f32_after.txt` logs are intentionally retained, but they are not
valid proof logs: RCH artifact retrieval repeatedly restored stale source into
the main worktree, so those runs filtered out the new test or did not execute
the new benchmark.

## Decision

Rejected/not kept in this pass. The runtime source hunks are not present in the
main worktree and are not committed under this bead. The target remains viable,
but it needs an uncontended ft-api/ft-kernel-cpu edit window or an isolated
worktree with source retrieval disabled.

Score: `0` for this pass because no after benchmark was produced from a stable
main-worktree candidate and no runtime change is kept.
