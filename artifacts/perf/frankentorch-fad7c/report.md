# frankentorch-fad7c RMSNorm rstd-stats rejection

Bead: `frankentorch-fad7c`

Target: `rms_norm/grad_2048x1024` in `ft-api` Criterion.

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-api --bench ops_bench -- rms_norm/grad_2048x1024 --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Initial worker `vmi1227854` baseline:

```text
rms_norm/grad_2048x1024 time: [117.17 ms 118.95 ms 120.76 ms]
```

Fresh paired worker `ovh-a` baseline:

```text
rms_norm/grad_2048x1024 time: [127.87 ms 128.97 ms 130.36 ms]
```

## Candidate

One lever tested: save the exact forward per-row RMS `rstd` sidecar and reuse it in RMSNorm backward, mirroring the existing `layer_norm_forward_with_stats_f64` pattern.

Proof during probe:

- `functional_rms_norm_grad_golden_bits` before/after golden bit lines matched.
- SHA-256: `5ea5df39620d5333d1a3e175ebaab8726963aed3a5db69cedc6955b77577907e`.
- `sha256sum -c artifacts/perf/frankentorch-fad7c/golden_after_sha256.txt` passed.
- Focused `ft-kernel-cpu` bit-exact recompute-vs-stats test passed during the probe.
- Focused `ft-api functional_rms_norm` tests passed during the probe.

After runs:

```text
vmi1227854 candidate: [111.23 ms 115.03 ms 119.09 ms]
ovh-a candidate:      [129.22 ms 129.86 ms 130.47 ms]
```

## Decision

Rejected. The first same-worker pair on `vmi1227854` showed only a modest overlapping-interval median win (`118.95 ms -> 115.03 ms`, `1.034x`). The fresh paired run on `ovh-a` did not reproduce it (`128.97 ms -> 129.86 ms`, `0.993x`).

Score: `0.0` keep score because the robust paired evidence failed the "real win" gate.

Source state: no RMSNorm source hunk is kept.

## Isomorphism Ledger

- Ordering preserved during probe: yes; row traversal and element traversal stayed unchanged.
- Tie-breaking: N/A.
- Floating-point: output, input grad, and weight grad golden bit lines matched exactly.
- RNG: unchanged; deterministic fixture only.
- Golden SHA verification: passed for the probe.

## Next Route

Do not repeat this RMSNorm stat-sidecar micro-lever. Re-profile ready perf beads and attack a deeper primitive, preferably `frankentorch-4iwpr` AdamW workspace/layout or another profile-backed structural kernel where the same-worker gate can clear `Score >= 2.0`.
