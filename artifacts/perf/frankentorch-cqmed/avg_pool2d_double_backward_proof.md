# cqmed: avg_pool2d double-backward — PARITY PROOF

## Before
- avg_pool2d f64 grad: ERR (tensor_apply_function, no create_graph backward)
- avg_pool2d f32 grad: silent ZERO Hessian (op-graph detachment) + f32->f64
  dtype-parity bug (grad output was F64 not F32)

## Fix
avg_pool2d is LINEAR (y = A·x, A geometry-only), so the create_graph adjoint
closes under its own kernels: dpadded = avg_pool2d_backward(dout), cotangent
grad_dout = avg_pool2d_forward(g). No saved tensors.
- f64: tensor_apply_function -> tensor_apply_function_with_create_graph + cg.
- f32: new fused grad path (apply_function_f32_output_with_create_graph_
  borrowed_inputs) + new avg_pool2d_backward_f32 kernel. Output now F32 (parity).

## After — input-Hessian diag of sum(avg_pool2d(x)^2), FrankenTorch == torch f64/f32
- nopad cip=true:  [0.125]*16   (both f64 + f32, exact)
- pad1  cip=true:  [0.125]*16   (exact)
- pad1  cip=false: [2.0,0.5,0.5,2.0, 0.5,0.125,0.125,0.5, ...]  (exact)

Probes: struct_hessian_probe, avgpool_gp_probe.
Tests: ft-kernel-cpu 483 passed; ft-api 2131 passed; 0 failed.
