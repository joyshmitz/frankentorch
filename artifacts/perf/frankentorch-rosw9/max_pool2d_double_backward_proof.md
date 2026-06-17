# rosw9: max_pool2d double-backward — PARITY PROOF

## Before
- max_pool2d f64 grad: ERR (tensor_apply_function, no create_graph backward)
- max_pool2d f32 grad: silent ZERO Hessian (op-graph detachment) + f32->f64 upcast

## Fix
max_pool2d is PIECEWISE-LINEAR (y = S·x, S = argmax selection, locally constant
in x). 1st-order backward dinput = Sᵀ·dout scatters dout to the argmax; the cg
adjoint GATHERS the cotangent g at the same argmax (grad wrt input = 0 a.e.,
matching torch).
- f64: tensor_apply_function -> tensor_apply_function_with_create_graph + cg
  (recompute argmax from input, scatter; backward gathers at saved indices).
- f32: new fused grad path (F32 output node — torch parity) + new
  max_pool2d_forward_with_indices_f32 / max_pool2d_backward_from_indices_f32.
  cg node casts the F32 input to F64 grad-space (gir5b).

## After — input-Hessian diag of sum(max_pool2d(x)^2), FT == torch (exact)
- f64: [0,0,0,0,0,2,2,0]   f32: [0,0,0,0,0,2,2,0]   (both match torch)

## After — full gradient-penalty (grad(sum(grad(sum(y^2),x,cg)^2), x)), exact
- f64: pen=43.3377 gx_sum=76.2320
- f32: pen=43.3377 gx_sum=76.2320

Probes: struct_hessian_probe, maxpool_gp_probe.
Tests: ft-kernel-cpu 483 passed; ft-api 2131 passed; 0 failed.
