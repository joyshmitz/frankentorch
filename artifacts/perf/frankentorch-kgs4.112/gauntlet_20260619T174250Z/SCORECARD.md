# frankentorch-kgs4.112 scorecard

- Agent: `IvoryDeer`
- Bead: `frankentorch-kgs4.112`
- Target: f64 avg_pool2d 2x2 stride-2 no-padding backward.
- Workload: `[N,C,H,W]=[8,64,64,64]`, scalar `sum`, backward.

## Performance

| Row | Worker/runtime | FrankenTorch | PyTorch | Ratio/verdict |
|---|---|---:|---:|---|
| Existing fast-path `ops_bench` baseline | rch `hz2` | `58.600 ms` | n/a | verified current row |
| Direct-assignment candidate | rch `hz2` | `68.624 ms` | n/a | rejected; `+13.137%`, `p=0.01` |
| Generic-disabled routing row | rch `ovh-b` | `117.51 ms` | n/a | cross-worker routing evidence only |
| Gauntlet head-to-head | local `/tmp/torchvenv/bin/python` | `16.627 ms` | `3.6632 ms` | `4.54x` slower, `0W / 1L / 0N` |
| Remote gauntlet FT arm | rch `hz2` | `13.383 ms` | unavailable | PyTorch missing `torch` on worker |

## Correctness And Gates

| Gate | Result |
|---|---|
| `rustfmt --edition 2024 --check crates/ft-api/benches/pytorch_gauntlet_bench.rs` | passed |
| `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench` | passed on `ovh-b` |
| `rch exec -- cargo test -p ft-kernel-cpu avg_pool2d_2x2s2_backward_matches_generic_bit_exact -- --nocapture` | passed on `vmi1264463` |
| `rch exec -- cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings` | passed on `hz2` |
| `rch exec -- cargo test -p ft-conformance` | passed locally via rch fallback: conformance green |
| `ubs` on changed Rust/Python/docs/artifact markdown files | zero critical or warning findings |
| `rch exec -- cargo fmt --check` | reports pre-existing unrelated formatting drift in `ft-api` examples; changed bench file direct rustfmt check passed |

## Decision

The existing 2x2s2 avg_pool2d backward specialization is verified, but the new
direct-assignment scatter variant regressed and was reverted. This closeout
keeps only the PyTorch gauntlet row and evidence artifacts; product source is
unchanged. The next viable gap-closing pass should target whole-workload
tape/allocation/sum-backward overhead or a fused training primitive, not another
local scatter micro-branch.
