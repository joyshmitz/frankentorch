# frankentorch-kgs4.156 — embedding_bag save-skip (don't clone the whole weight table for sum/mean grad)

Date: 2026-06-21
Agent: cc
Second application of the kgs4-diagnosis save-skip/borrowed lever (after kgs4.155 SDPA).

## Lever

`tensor_embedding_bag`'s f64 grad path did `ctx.save_for_backward(weight_vals.to_vec(), ...)`
unconditionally — cloning the ENTIRE `[num_embeddings, embedding_dim]` weight table into ctx
every forward. But only the "max" backward needs the weight (to recompute the per-column
argmax); "sum"/"mean" backward use only indices/offsets/grad. So for the common sum/mean modes
that was a dead clone of the whole embedding table per step. Gated the save on `mode=="max"`
(and moved the `weight_vals` bind into the max branch). Bit-exact; sum/mean unaffected, max
unchanged.

## Correctness (bit-exact)

Same-process A/B reports IDENTICAL grad checksum `3.932160e5` always-save vs save-skip. ft-api
`--lib embedding_bag` 2 passed / 0 failed; ft-conformance green.

## Measurement (same-host, f64 embedding_bag SUM train, vocab=50000 dim=128 bags=256 bag=12, 32 threads)

Worker-immune fixed-iter harness (`example embedding_bag_retention_ab`, 150 reused-session
fwd+sum+bwd+grad-read steps); PyTorch `F.embedding_bag(..., mode='sum').sum().backward()`:

| Path | ms/step | vs PyTorch | vs baseline |
| --- | ---: | --- | --- |
| PyTorch f64 | `11.0` | — | — |
| FT always-save (baseline) | `~75` | 6.8x slower | — |
| **FT save-skip (this lever)** | **`~45`** | 4.1x slower | **1.68x faster (bit-exact)** |

## Verdict: KEEP — internal 1.68x, PyTorch loss (4.1x)

Internal-keep / PyTorch-loss (same disposition as kgs4.138-145): the dead full-table clone is
gone (helps ALL sum/mean embedding_bag training, not just this shape), but FT is still 4.1x
slower than PyTorch. The residual gap is the **dense `grad_weight` buffer** — FT zeroes a full
`num_embeddings*embedding_dim` (51 MB here) dense gradient per step, while PyTorch returns a
SPARSE/indexed grad. Closing it needs a sparse embedding-grad representation (a separate, bigger
lever), exactly the "per-step dense-buffer allocation" the kgs4 cross-cutting diagnosis names.

## Win/loss/neutral vs PyTorch (32t): `0W / 1L / 0N` (internal 1.68x kept)

## Gates
- `cargo test -p ft-api --release --lib embedding_bag`: 2 passed, 0 failed.
- `cargo test -p ft-conformance --release`: green.
