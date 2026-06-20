# frankentorch-kgs4.118 Scorecard

## Lever

Existing code-first lever from commit `75d87600`: specialize f64 `conv3d_backward_f64`
when the upstream `dout` slice is non-empty and exactly all `+1.0`, which is the
`tensor_sum(out).backward()` training case in `ops_bench` `conv3d/grad`.

The fast path collapses repeated all-ones matrices into one-row reductions and a
repeated-row col2im scatter. Non-unit or empty `dout` stays on the generic path.

## Verdict

KEEP as a measured FrankenTorch internal win.

This does not dominate PyTorch. The same-shape local PyTorch CPU row is still `3.50x`
faster than current FrankenTorch, so the release ledger records this as a PyTorch loss.

## Head-to-head

| Case | Median | Result |
| --- | ---: | --- |
| FrankenTorch parent `870abe0d` on `ovh-a` | 29.723 ms | baseline |
| FrankenTorch current `main` on `ovh-a` | 26.595 ms | WIN, 1.12x faster / -10.5% |
| Local PyTorch 2.12.1+cpu, same f64 shape | 7.593859 ms | PyTorch wins |
| Current FT / PyTorch ratio | 3.50x slower | LOSS remains |

PyTorch W/L/N for this row: `0 / 1 / 0`.

## Commands

Baseline:

```text
RCH_WORKER=ovh-a RCH_WORKERS=ovh-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a \
rch exec -- cargo bench -p ft-api --bench ops_bench -- conv3d/grad --noplot
```

Current:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a \
rch exec -- cargo bench -p ft-api --bench ops_bench -- conv3d/grad --noplot
```

PyTorch:

```text
/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python - <<'PY'
```

The PyTorch script used `torch.set_num_threads(32)`,
`torch.set_num_interop_threads(32)`, f64 input shape `[2,32,8,16,16]`,
f64 weight shape `[32,32,3,3,3]`, stride `(1,1,1)`, padding `(1,1,1)`, scalar
sum loss, and backward over 40 timed samples after warmup.

## Validation

| Gate | Result |
| --- | --- |
| `rch exec -- cargo test -p ft-kernel-cpu conv3d --lib` | PASS, 2/0 |
| `rch exec -- cargo test -p ft-api conv3d --lib` | PASS, 10/0 |
| `rch exec -- cargo test -p ft-conformance strict_scheduler_conformance_is_green --lib` | PASS, 1/0 |

## Follow-up Routing

The next conv/matmul work should attack the remaining end-to-end gap, not retry this
ones-`dout` collapse. Current evidence points at PyTorch still being about `3.50x`
faster for this f64 conv3d training row.
