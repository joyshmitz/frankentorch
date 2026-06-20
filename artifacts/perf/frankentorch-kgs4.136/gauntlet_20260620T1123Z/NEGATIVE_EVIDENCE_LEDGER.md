# frankentorch-kgs4.136 Negative-Evidence Summary

Workload: f32 BatchNorm2d train step, `[N,C,H,W]=[32,256,28,28]`, affine grads, scalar `sum` loss.

Lever: `functional_batch_norm2d_sum` plus f32 scalar forward/backward helpers, avoiding materialized normalized output and dense all-ones `dy`.

Results:

- Baseline diagnostic, rch `ovh-a`, `[16,64,28,28]`: composed `129.84 ms`, existing fused `13.59 ms`.
- Candidate diagnostic, rch `vmi1227854`, `[16,64,28,28]`: composed `109.59 ms`, existing fused `10.80 ms`, scalar `1.66 ms`; scalar/fused `0.1537x`, `6.50x` faster.
- Target Criterion, rch `vmi1227854`, `[32,256,28,28]`: existing fused mean `114.23 ms`, scalar mean `78.166 ms`; scalar/fused `0.6843x`, `1.46x` faster.
- Remote PyTorch arm failed on rch: `ModuleNotFoundError: No module named 'torch'`.
- Local PyTorch oracle: 30 iterations in `0.168172072968 s`, `5.605736 ms/iter`; scalar/PyTorch `13.94x` slower.

Verdict: keep as an internal win and PyTorch loss (`0W / 1L / 0N`).

Retry boundary: do not retry the rejected dense all-ones `dy` branch or another wrapper that only removes `tensor_sum`. Move deeper into stats/backward reuse, arena/tape allocation, automatic scalar-loss fusion, f32 storage/layout, or PyTorch-parity proof for algebraically zero BatchNorm sum-loss gradients.

Static note: scoped UBS timed out after 240 seconds with no findings emitted beyond `Scanning rust...`; kernel scalar-backward unit-upstream and scaled-upstream tests, API tests, compile, clippy, conformance, targeted rustfmt, and diff whitespace gates passed.
