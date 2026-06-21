# frankentorch-kgs4.155 — borrowed-inputs for the f64 SDPA grad path (scaled_dot_product_attention entry) — TRAIN-STEP WIN

Date: 2026-06-21
Agent: cc
Direct application of the kgs4-diagnosis cheapest lever (#1 borrowed-inputs) to a specific
target: the closest train-step "loss" on record, kgs4.113 (SDPA f64 train, "1.29x slower").

## Lever

The public `scaled_dot_product_attention` (the 5083 entry) f64 grad fast paths — unmasked and
masked — used `tensor_apply_function` with `ctx.save_for_backward(qv.to_vec()/kv/vv)`, i.e. they
**cloned all three q/k/v tensors into the ctx every training step** (3×`numel` = ~12 MB/step at
`[16,512,64]`). The sibling `tensor_scaled_dot_product_attention` (19166 entry) already used
`tensor_apply_function_f64_borrowed_inputs` (frankentorch-3jmy3); the 5083 entry was missed.
Converted both 5083 f64 grad paths to borrowed-inputs: `sdpa_backward[_masked]_f64` re-reads
q/k/v from the live leaves instead of cloning them into ctx. The backward recomputes attention
internally, so nothing derived needs saving.

## Correctness (bit-exact)

Gradients are byte-identical (the backward reads the same q/k/v values, borrowed not cloned):
same-process A/B reported the IDENTICAL checksum `4.038907e4` for clone vs borrowed. ft-api
`--lib sdpa` 18 passed / 0 failed (incl. grad tests); ft-conformance green.

## Measurement (same-host, f64 SDPA train step `[16,512,64]`, 32 torch threads)

Worker-immune fixed-iter harness — q/k/v leaves built ONCE, 200 reused-session
forward+`sum()`+backward+grad-read steps, `example sdpa_f64_train_retention_ab`; PyTorch via
`benches/pytorch_sdpa_grad.py` (50 iters, 32 threads, same shape):

| Path | ms/step | vs PyTorch | vs clone baseline |
| --- | ---: | --- | --- |
| PyTorch f64 fwd+bwd | `44.1–45.2` | — | — |
| FT clone (baseline) | `27.5` | FT 1.6x faster | — |
| **FT borrowed (this lever)** | **`20.1–20.4`** | **FT 2.18–2.25x faster** | **1.35x faster (bit-exact)** |

Both reused-session runs were FLAT (late/early `0.98–1.01` over 200 steps) — no tape-retention
degradation at this scale. The FT harness even does MORE than PyTorch's (it reads the q-grad out
to host), so the 2.2x is conservative.

## Reconciliation with kgs4.113 ("SDPA f64 train 1.29x slower")

kgs4.113 measured via the criterion gauntlet bench, which is the documented criterion
**non-stationarity trap** for a tape-growth workload (one reused session across criterion's many
samples → accumulation skews the median). Under a fair fixed-iter harness FT is 2.2x FASTER, and
this lever widens that. The kgs4.113 "loss" was a measurement artifact, not a real regression.

## Win/loss/neutral vs PyTorch (32t): `1W / 0N` — first TRAIN-STEP win, validates the diagnosis

The cross-cutting diagnosis (NEGATIVE_EVIDENCE 2026-06-21) said per-step autograd allocation
floors the train-step frontier and named borrowed-inputs as the cheap bit-exact lever. This
confirms it: eliminating 3 clones/step turned the closest train-step case into a measured win.
The SAME audit should be applied to other `tensor_apply_function` sites that `save_for_backward`
full-size INPUT tensors (807/7698/8977/9926/... — next).

## Gates
- `cargo test -p ft-api --release --lib sdpa`: 18 passed, 0 failed.
- `cargo test -p ft-conformance --release`: green.
