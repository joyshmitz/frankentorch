# lboou: conv3d-f32 + conv_transpose2d-f32 double-backward — PARITY PROOF

## Before (bugs)
- conv3d_f32 input-Hessian: ERR DenseTensor(UnsupportedDType(F32))
  (f32 grad fell through to narrow/cat/bmm op-graph; no f32 2nd-order)
- conv_transpose2d_f32 input-Hessian: diag=[0,0,0,0] (ALL ZERO — silent
  detachment in the f32 narrow/matmul/pad/add scatter)

## Fix
New f32 kernels (ft-kernel-cpu): conv3d_col2im_f32, conv3d_backward_f32,
conv_transpose2d_backward_f32 (ports of the f64 forms, sgemm/f32 math).
New fused f32 GRAD paths (ft-api): mirror the f64 conv3d/conv_transpose2d
custom ops via apply_function_f32_output_with_create_graph_borrowed_inputs;
bilinear create_graph adjoints close under the op's own f32 forward/backward;
grad NODES cast to F64 grad-space (gir5b recipe) so f64-only readers accept them.

## After — input-Hessian diag (FrankenTorch == torch 2.12 f32, exact)
- conv3d_f32:           [0.32, 0.4, 0.08, 0.32, 0.48, 0.16, 0.0, 0.08, 0.08]
- conv_transpose2d_f32: [0.6, 0.6, 0.6, 0.6]

## After — full gradient-penalty (input+weight+bias double-backward), exact
- conv3d_f32:           pen=0.2207 gw_sum=-0.8576 gb=[1.2928]
- conv_transpose2d_f32: pen=0.1242 gw_sum=-0.1738 gb=[0.1088]

Probes: conv3d_ct2d_f32_hessian_probe, conv3d_ct2d_f32_gp_probe.
Tests: ft-kernel-cpu 483 passed; ft-api 2131 passed; 0 failed.
