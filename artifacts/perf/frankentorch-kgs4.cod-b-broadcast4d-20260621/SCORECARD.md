# Broadcast-H masked f64 SDPA proof

Date: 2026-06-21
Assignee: cod-b
Agent: IvoryDeer
Target dir: `/data/projects/.rch-targets/frankentorch-cod-b`

## Verdict

No-ship. The broadcast-over-heads mask path `[B,1,S,S]` is numerically correct, but the same-host PyTorch
comparison shows PyTorch is faster on the standard 4-D layout. The measurement-only example row was
reverted; no product source is kept from this probe.

The radical lever tested here was a mask-stride/broadcast-head fast path: avoid materializing
`B*H*S*S` mask storage by indexing the broadcast mask directly during masked SDPA. That looked promising
from cache and allocation pressure, but it does not beat PyTorch for this shape.

## Same-host ratio source

Command source: retrieved warm-target FrankenTorch release example binary plus local PyTorch CPU.

Log: `local_binary_sdpa_masked_broadcast4d.log`

| Lane | FrankenTorch | PyTorch | Ratio | Correctness |
| --- | ---: | ---: | ---: | --- |
| primary masked f64 SDPA, 3-D | 7.868 ms | 18.801 ms | FT 2.39x faster | rel-diff 3.29e-14 |
| tensor masked f64 SDPA, 3-D | 8.545 ms | 19.360 ms | FT 2.27x faster | rel-diff 3.29e-14 |
| masked f64 GQA | 51.332 ms | 5.435 ms | FT 9.44x slower | rel-diff 3.19e-14 |
| broadcast-H masked f64 SDPA, 4-D | 7.518 ms | 5.430 ms | FT 1.38x slower | rel-diff 1.70e-14 |

Scorecard for this proof bundle: 2W / 2L / 0N overall. Broadcast-H specifically: 0W / 1L / 0N.

## RCH evidence

RCH FT-only run: `rch_sdpa_masked_broadcast4d.log`

The remote worker did not have the configured PyTorch interpreter, so the RCH run is FrankenTorch routing
evidence only, not the ratio source.

Observed RCH FT-only timings:

- primary: 10.313 ms
- tensor: 10.297 ms
- GQA: 22.157 ms
- broadcast-H 4-D: 10.733 ms

## Conformance

Gate: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b AGENT_NAME=IvoryDeer rch exec -- cargo test -p ft-conformance --profile release`

Log: `test_ft_conformance_release.log`

Result: green on `vmi1153651`; 199 `ft_conformance` lib tests plus conformance bins, integration tests,
smoke tests, and doctests passed.

## Retry predicate

Do not ship the broadcast-H mask-div path as a PyTorch-performance lever for this 4-D shape. The next
credible target is a direct grouped/GQA masked f64 kernel that indexes `kv_head = q_head / group` without
expanding K/V heads, then reruns the same head-to-head scorecard.
