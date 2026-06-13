# frankentorch-kgs4.69 - safe-Rust f32 sgemm_bt microkernel rejection

## Target

- Bead: `frankentorch-kgs4.69`
- Hotspot: `linear_forward/f32_hidden/{1024,2048}` -> `ft-kernel-cpu::gemm::sgemm_bt`
- Profile lineage: successor to rejected f32 Linear borrowed-input and per-call packed-panel passes.
- Alien-graveyard primitive: dense submatrix BLAS-3 inner kernels with tiling/register blocking.

## Baseline

RCH worker: `vmi1149989`

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-api --bench ops_bench -- linear_forward/f32_hidden --sample-size 20 --warm-up-time 1 --measurement-time 3
```

Criterion medians:

- `linear_forward/f32_hidden/1024`: `[447.52 us 474.27 us 499.02 us]`
- `linear_forward/f32_hidden/2048`: `[682.99 us 730.92 us 771.28 us]`

## One Lever Attempted

Shape-gated candidate for `m=32, k=512, n in {1024,2048}`:

- Pure safe Rust inside the existing GEMM module.
- Parallelized over output rows.
- Four-column register tile per inner loop.
- Per-output accumulation used a single `kk=0..512` `mul_add` chain.

## Behavior Proof

Candidate proof command:

```text
RCH_WORKER=vmi1149989 RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-kernel-cpu --lib sgemm_bt_m32k512_safe_microkernel_matches_matrixmultiply_bits -- --nocapture
```

Result: FAILED before candidate benchmarking.

Failure:

```text
safe m32k512 sgemm_bt diverged for n=1024
```

Isomorphism:

- Ordering preserved: no, candidate changes the existing `matrixmultiply` accumulation tree into a scalar row-local chain.
- Tie-breaking unchanged: N/A for dense GEMM, no comparisons.
- Floating-point: not identical; strict `f32::to_bits` proof failed.
- RNG seeds: unchanged; benchmark and golden test constructors unchanged.
- Golden output after source removal: `functional_linear_f32_fused_matches_transpose_path_bit_exact` passed via RCH.

Restored-path golden command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-api --lib functional_linear_f32_fused_matches_transpose_path_bit_exact -- --nocapture
```

Result: passed `1/1` after removing the source hunk.

## Artifact SHA-256

```text
d61d729a0579b9bd83310fb068aca2587aec4ba957585d5a25921575f88076d9  artifacts/perf/frankentorch-kgs4.69/baseline_linear_f32_hidden.log
118db4ef6f2de246dc84cbbf6a6c0a254389bb79b610a32bc16f8d2e3428286d  artifacts/perf/frankentorch-kgs4.69/candidate_bitproof.log
413c3cd048f4023afa5230e52ff865d515adcbf7e3365c6825aa8cd6fb732747  artifacts/perf/frankentorch-kgs4.69/restored_f32_linear_golden.log
```

## Verdict

Rejected. Score `0.0` because the candidate failed behavior proof before benchmarking.

No production source is retained. `crates/ft-kernel-cpu/src/lib.rs` has no final diff for this bead.

## Next Primitive

Do not repeat scalar row-local dot or per-call B packing. The next `sgemm_bt` route should attack an accumulation-order-compatible safe-Rust primitive:

- first derive the existing `matrixmultiply` f32 BT accumulation tree for the target shape,
- build a shadow microkernel that reproduces that tree exactly under `to_bits`,
- then register-tile or block only after the shadow kernel proves bit identity.

Target ratio: `>= 1.20x` on `linear_forward/f32_hidden/{1024,2048}` same-worker Criterion, with strict f32 fused-linear golden preserved.
