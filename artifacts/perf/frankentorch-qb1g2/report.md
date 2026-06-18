# frankentorch-qb1g2 - deterministic safe-Rust GEMM lane

## Target

- Bead: `frankentorch-qb1g2`
- Selection: no ready `[perf]` bead appeared after a brief wait; active perf lanes are claimed by other agents. This ready bug is profile/root-cause backed: matmul-backed kernels depend on `matrixmultiply` runtime CPU dispatch, whose FMA vs non-FMA accumulation changes low bits across workers.
- Coordination: claimed by `RubyLotus`; reserved `crates/ft-kernel-cpu/src/lib.rs`, `crates/ft-kernel-cpu/benches/*`, this artifact directory, `.skill-loop-progress.md`, and `.beads/issues.jsonl`.

## Baseline

Initial remote baseline command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench gemm_bench -- 'matmul_.*_512x512x512' --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Result: blocked before compilation. RCH refused local fallback because no admissible worker was available.

Second all-512 attempt on `ts2` produced only build output and no Criterion measurements, so it is not a valid baseline.

Retained f64 baseline:

```bash
RCH_REQUIRE_REMOTE=1 RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR CARGO_TARGET_DIR=/data/tmp/frankentorch-qb1g2-target \
  rch exec -- cargo bench -p ft-kernel-cpu --bench gemm_bench -- matmul_f64_512x512x512 \
  --warm-up-time 1 --measurement-time 3 --sample-size 12 --noplot
```

- Worker: `vmi1167313`
- `matmul_f64_512x512x512`: `[7.6685 ms 8.3931 ms 9.3803 ms]`

Secondary f32 baseline on `ts2` again exited without Criterion measurements; it is recorded as a failed secondary attempt and is not used for scoring.

## Candidate Primitive

One lever under test: deterministic safe-Rust f64 `mul_add` GEMM for the benchmark/root-cause square block. The current hunk fixes the output-cell order and accumulates each dot product with ascending `p = 0..k`.

## Isomorphism Obligations

- Ordering: output cells are visited deterministically by row/column; each dot product accumulates `p = 0..k` in ascending order.
- Floating point: every product contribution uses explicit single-rounding `mul_add`; any changed rounding must be justified as the intended fix for cross-worker FMA drift and checked against golden outputs.
- Tie-breaking: GEMM has no data-dependent ordering or ties.
- RNG: no RNG involved.
- Shapes/errors/storage offsets: public matmul, bmm, addmm validation and error surfaces remain unchanged.
- Golden SHA: run `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` after the candidate.
