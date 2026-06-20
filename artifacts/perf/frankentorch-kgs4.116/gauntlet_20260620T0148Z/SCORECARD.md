# frankentorch-kgs4.116 scorecard

## Summary

- Bead: `frankentorch-kgs4.116`
- Lane: LayerNorm unit-dy backward fast path
- Result: measured internal keep, PyTorch loss
- Agent: `IvoryDeer` / `cod-a`
- Source status: no new product source edits in this closeout; verified the
  already-landed code-first commit `1e6af1d8`.

## Performance

| Comparison | Worker | Result | Verdict |
| --- | --- | ---: | --- |
| Parent `2aa78200` `layer_norm/grad_2048x1024` | `hz2` | `90.723 ms` | baseline |
| Current `45c2e011` `layer_norm/grad_2048x1024` | `hz2` | `29.606 ms` | `3.06x` faster |
| PyTorch f64 CPU local oracle | local | median `8.261743 ms` | FT is `3.58x` slower |
| f32 composed vs fused diagnostic `[8192,1024]` | `hz2` | `1930.66 ms -> 293.49 ms` | `6.58x` faster |

## Correctness Gates

| Gate | Result |
| --- | --- |
| `rch exec -- cargo test -p ft-kernel-cpu layer_norm_f -- --nocapture` | passed: 2/0 unit-dy bit-exact tests |
| `rch exec -- cargo test -p ft-api functional_layer_norm -- --nocapture` | passed: 7/0 API LayerNorm tests |
| `rch exec -- cargo test -p ft-conformance strict_scheduler -- --nocapture` | first worker `vmi1264463` canceled for stale progress; retry on `hz2` passed: 1/0 |

## Negative Evidence

- PyTorch remains faster: `3.58x` by median for the matched f64 CPU oracle row.
- Remote PyTorch could not run because worker Python lacks `torch`.
- The f32 example is strong supporting evidence but not a PyTorch ratio and not
  the official bead benchmark.

## Release Readiness

Keep the fast path as a proven same-worker internal win. The release scorecard
must count this as a PyTorch loss, not a dominance row. Route remaining work to
whole-row overhead: allocation/tape/loss fusion, persistent workspaces,
deterministic parallel affine reductions, and f32-native end-to-end training
rows.
