# frankentorch-fql10 pass 2 multishift/AED contract

Date: 2026-06-11
Agent: BlackThrush
Bead: frankentorch-fql10

## Scope

This pass updates the parent fql10 route after finishing the ct2yy QR lane. It
does not change source behavior. The purpose is to bind the next source lever to
fresh remote baseline evidence and the already-rejected qglh3 AED variants.

## Fresh remote baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -v -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- '^(eigvals_f64_256x256|eig_f64_256x256)$' --sample-size 10 --warm-up-time 1 --measurement-time 3
```

RCH selected `vmi1227854`.

| Row | Estimate |
| --- | ---: |
| `eig_f64_256x256` | `[74.376 ms 79.126 ms 86.169 ms]` |
| `eigvals_f64_256x256` | `[26.601 ms 27.645 ms 28.398 ms]` |

Log SHA-256:

```text
98310280dd50127c26346b4c52a7eaa470d0acb70492f8b905da7f4b36f5f075  artifacts/perf/frankentorch-fql10/pass2_remote_baseline_eig_eigvals_256.log
```

The full `eig` row is noisy relative to prior same-worker qglh3 evidence, but
the `eigvals` row remains stable around `27 ms`. That keeps the shared serial
Francis QR floor as the target, not back-substitution or q_acc replay.

## Golden state

Remote golden retries were refused by RCH admission
(`critical_pressure=1,insufficient_slots=1`). The pass-1 deterministic golden
stdout was rechecked locally:

```text
24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725  artifacts/perf/frankentorch-fql10/pass1_eigvals_golden_stdout.txt
```

Printed fixture digests remain the current strict fallback contract:

| n | `eigvals_digest` | `eig_digest` |
| ---: | --- | --- |
| 64 | `0xbc0583d464b1a211` | `0xbc0583d464b1a211` |
| 128 | `0x763c4b15d92c4b89` | `0x763c4b15d92c4b89` |
| 256 | `0x00b87b4996340204` | `0x00b87b4996340204` |

Any source pass must regenerate this after the edit. If RCH refuses a remote
golden, the source pass can use local golden only as fallback proof, never as
same-worker performance evidence.

## Rejected families

Do not repeat these:

- eigenvector back-substitution or q_acc-only work: already Amdahl-capped and
  not the shared `eigvals` floor;
- active-column/window trims: prior dense spectra keep `l` near zero;
- values-only AED suffix deflation: qglh3 final retry regressed
  `27.445 ms -> 29.767 ms`;
- full-window all-or-nothing AED with q_acc: qglh3 candidate regressed
  `49.892 ms -> 74.874 ms` against the stable full-eig comparator.

## Selected primitive

The next source lever must be structurally different:

1. If continuing qglh3: partial AED window record with Schur-block reordering
   and explicit undeflated shifts, not whole-window threshold deflation.
2. If opening npxbw: direct small-bulge multishift scaffolding that consumes an
   explicit shift list and can batch far row/column updates through the existing
   safe-Rust GEMM path.

The parent fql10 target remains LAPACK-style `dlaqr0`: size-gated current
double-shift fallback for small or strict cases, AED for deflation and shifts,
then multishift `dlaqr5`-style sweeps for BLAS-3 throughput.

## Proof contract

- Ordering: preserve interleaved `(re, im)` public order and conjugate adjacency.
- Ties/near-degeneracy: use strict current double-shift fallback unless the AED
  window residual and spike bound prove a safe split.
- Floating point: public vector outputs may use the qgce4 tolerance policy; the
  strict fallback path must keep the current golden digests.
- RNG: none.
- Reconstruction: for full `eig`, prove `A V ~= V Lambda` and q_acc/window
  orthogonality before dispatch changes.
- Performance: compare against the `vmi1227854` `eigvals_f64_256x256` median
  `27.645 ms` for shared QR work, and use full `eig` only after a clean
  same-worker comparator is available.

## Score gate

Expected score for the next concrete source lever:

| Candidate | Impact | Confidence | Effort | Score | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| partial AED shift-list record with block reordering | 4 | 3 | 3 | 4.0 | proceed in qglh3 if unclaimed |
| npxbw multishift scaffolding with fixed synthetic shifts | 5 | 2 | 4 | 2.5 | proceed after shift list exists |
| more threshold-only AED | 0 | 1 | 1 | 0.0 | reject |

Pass 3 should either claim the specific child bead that owns the edit surface or
stay evidence-only. The parent fql10 bead should not receive another ambiguous
runtime hunk.
