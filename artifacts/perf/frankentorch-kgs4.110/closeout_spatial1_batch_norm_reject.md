# frankentorch-kgs4.110 closeout - spatial1 BatchNorm row-major reduction attempt

## Decision

REJECT.

The candidate preserved behavior but did not improve the hot row. The source
change was removed; no BatchNorm kernel change is kept for this bead.

## Profile-backed target

- Source profile: `artifacts/perf/frankentorch-next-reprofile-20260617/current_top_train_reprofile.log`
- Source profile SHA-256:
  `a5ee5493acf5315e43b80c6a8a8d4a4ac4fd5d8beada61c1abb8f68a5880bea3`
- Hot row: `batch_norm/grad_1d_8192x1024` `[678.28 ms 693.66 ms 717.16 ms]`
- Dedicated local baseline:
  `artifacts/perf/frankentorch-kgs4.110/pass1_local_baseline_batch_norm1d_grad.log`
  reported `[689.25 ms 701.11 ms 712.43 ms]`
- Baseline artifact SHA-256:
  `76ac7adbfe89f5ca762bad4701ebe4809355a1917181f3af4b77b9e91fa07b61`

## One lever

The rejected candidate specialized f64 BatchNorm `spatial == 1` stats/backward
reductions for row-major `[N,C]` channel blocks instead of one-channel strided
scans.

## Isomorphism proof

- Per-channel sample order preserved: every channel still accumulated `n` from
  `0..batch` ascending.
- Floating point formula preserved: mean, variance, affine output, `dweight`,
  `dbias`, and `dx` formulas were unchanged.
- Running-stat and Bessel behavior unchanged: no API-level running-stat code was
  changed.
- RNG preserved: no RNG is involved.
- Golden proof:
  `artifacts/perf/frankentorch-kgs4.110/pass2_kernel_proof_batch_norm_spatial1.log`,
  SHA-256 `64298625cdf4bf0600ccd5c39fef40eb8d53626027572b4c980178f8c85add5e`

## Rebench

Final local Criterion rebench:

- Artifact: `artifacts/perf/frankentorch-kgs4.110/pass3c_local_rebench_batch_norm1d_grad.log`
- Candidate: `[686.28 ms 701.45 ms 716.41 ms]`
- Baseline median to candidate median: `701.11 ms -> 701.45 ms`
- Criterion result: change `[-2.6571% +0.0491% +2.9602%]`, `p = 0.97`

Score:

- Impact: `0.00`
- Confidence: `0.98`
- Effort: `0.60`
- Score: `0.00`

## Outcome

No source change was kept. This rejected micro-layout attempt routes the next
pass to a different primitive rather than another spatial1 BatchNorm reduction
variant.
