# frankentorch-kgs4.110 closeout

Decision: REJECT.

Profile-backed target:
- Source: `artifacts/perf/frankentorch-next-reprofile-20260617/current_top_train_reprofile.log`
- Row: `batch_norm/grad_1d_8192x1024` `[678.28 ms 693.66 ms 717.16 ms]`

Dedicated local baseline:
- Command: `RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/tmp/frankentorch-next-reprofile-local-target cargo bench -j 1 -p ft-api --bench ops_bench -- 'batch_norm/grad_1d_8192x1024' --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot`
- Result: `[689.25 ms 701.11 ms 712.43 ms]`
- Artifact: `artifacts/perf/frankentorch-kgs4.110/pass1_local_baseline_batch_norm1d_grad.log`
- SHA-256: `4c6eab967aec3e9237a426d5164ab8fe8b55566bd4fafada0e7fc7426c1e21ce`

Candidate lever:
- Prototype: spatial-1 f64 BatchNorm row-major/channel-chunk reduction path.
- Invariants required: per-channel `n`-ascending accumulation order, variance formula, running-stat Bessel update, affine semantics, backward `dx/dweight/dbias` math, dtype/error behavior, and RNG absence.

Behavior proof:
- Focused proof: `cargo test -j 1 -p ft-kernel-cpu batch_norm_f64_spatial1_row_parallel_matches_serial_reference_bits --lib -- --nocapture`
- Result: pass.
- Artifact: `artifacts/perf/frankentorch-kgs4.110/pass2_kernel_proof_batch_norm_spatial1.log`

Rebench:
- Same-target result: `[694.45 ms 706.74 ms 718.14 ms]`
- Criterion delta: `[-1.9278% +0.7545% +3.6789%]`, `p = 0.61`; no significant change detected.
- Artifact: `artifacts/perf/frankentorch-kgs4.110/pass3b_local_rebench_batch_norm1d_grad_same_target.log`

Score:
- `0.83 = Impact 0.99 * Confidence 0.84 / Effort 1.00`
- Below the `2.0` keep threshold.

Closeout:
- Source hunk removed.
- Do not repeat this row-major reduction micro-lever.
- Next BatchNorm route should be structural: save forward mean/invstd sidecars through the autograd path, or fuse BatchNorm affine backward with upstream gradient consumers where profiling shows the reduction remains dominant.
