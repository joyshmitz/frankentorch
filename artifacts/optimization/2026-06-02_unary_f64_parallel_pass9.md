# Unary F64 Parallel Output Pass 9

Bead: `frankentorch-clt9`

## Target

Profile-backed fallback target after pass 7:

- 2026-06-01 scenario: `exp/elements/1000000` measured [4.0370 ms 4.0406 ms 4.0452 ms].
- Fresh rch baseline: `exp/elements/1000000` measured [1.1590 ms 1.2372 ms 1.2882 ms] on worker `vmi1149989`.

The active conv2d/unfold pass owned `ft-api`, so this pass stayed in `ft-kernel-cpu`.

## Lever

One lever tested in the large contiguous `unary_f64` branch:

- Replace `window.par_iter().map(|value| op(*value)).collect()` with a pre-sized output
  buffer and parallel mutable writes.

## Behavior Proof

Proof while the lever was applied:

```text
sha256sum -c artifacts/optimization/golden_checksums.txt
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-clt9-target rch exec -- cargo test -p ft-kernel-cpu exp_tensor_contiguous -- --nocapture
```

The first test attempt exposed a test-only missing `crate::PARALLEL_THRESHOLD`
qualifier. After fixing that, the focused exp test filter passed.

Golden sha256:

```text
655a2cf7931e4ce19a8192234c93d0fdd2eed5e306e5e7c2d01da3758d7a2f6f  artifacts/optimization/golden_outputs/unary_f64_parallel_pass9.txt
```

Isomorphism notes:

- Ordering and tie-breaking: output indices matched input indices.
- Floating point: each element still received exactly one `f64::exp` call.
- Storage offsets: the temporary test covered an offset input spanning the parallel threshold.
- RNG: not involved.
- Diagnostics: validation branches were unchanged.

## Re-benchmark

Command:

```text
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-clt9-target rch exec -- cargo bench -p ft-api --bench ops_bench -- exp/elements/1000000 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

After result on worker `ts2`:

```text
exp/elements/1000000 time: [9.7384 ms 10.150 ms 10.628 ms]
```

The after result was a clear regression versus the fresh baseline and large enough
that no same-worker control was needed to reject the lever.

## Verdict

Rejected. Score: impact -2 x confidence 2 / effort 1 = -4.0.

The kernel code and temporary test were reverted manually. The golden/evidence
artifacts were retained for audit.
