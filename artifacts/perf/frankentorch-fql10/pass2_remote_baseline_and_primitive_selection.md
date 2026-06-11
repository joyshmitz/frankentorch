# frankentorch-fql10 pass 2: remote baseline reset and primitive selection

## Scope

Bead: `frankentorch-fql10`

This pass refreshes Pass 1's local-fallback routing evidence with a real remote
RCH baseline and then selects the next one-lever source target. No runtime source
changed in this pass.

## Remote Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -v -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- '^(eigvals_f64_256x256|eig_f64_256x256)$' --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Worker: `vmi1227854`

| Row | Estimate |
| --- | ---: |
| `eig_f64_256x256` | `[74.376 ms 79.126 ms 86.169 ms]` |
| `eigvals_f64_256x256` | `[26.601 ms 27.645 ms 28.398 ms]` |

Baseline log SHA-256:

```text
98310280dd50127c26346b4c52a7eaa470d0acb70492f8b905da7f4b36f5f075  artifacts/perf/frankentorch-fql10/pass2_remote_baseline_eig_eigvals_256.log
```

Interpretation:

- Values-only `eigvals` is stable against the qglh3 baseline (`27.445 ms` median
  there, `27.645 ms` here).
- Full `eig` is much slower on the live tree (`79.126 ms` median). This reinforces
  that fql10 must improve the shared Francis-QR/AED machinery and the vector path
  together; another values-only shortcut is not sufficient.

## Golden Anchor

Remote golden attempts were refused before execution after the baseline:

- `pass2_eigvals_golden_remote.log`: shell-wrapped command rejected as non-compilation.
- `pass2_eigvals_golden_remote_direct.log`: no admissible worker.
- `pass2_eigvals_golden_remote_direct_retry.log`: no admissible worker after `rch workers probe --all`.

The strict behavior anchor remains the Pass 1 golden stdout:

```text
24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725  artifacts/perf/frankentorch-fql10/pass1_eigvals_golden_stdout.txt
```

Fixture digests:

| n | `eigvals_digest` | `eig_digest` |
| ---: | --- | --- |
| 64 | `0xbc0583d464b1a211` | `0xbc0583d464b1a211` |
| 128 | `0x763c4b15d92c4b89` | `0x763c4b15d92c4b89` |
| 256 | `0x00b87b4996340204` | `0x00b87b4996340204` |

## Canonical Primitive Harvest

Canonical graveyard mapping:

- `alien_cs_graveyard.md` section 9.6: communication-avoiding dense linear
  algebra, especially tree/block structure that converts scalar communication
  into larger local kernels with explicit stability/proof budgets.
- High-level FrankenSuite optimization contract: every performance change needs
  profile evidence, one lever, golden checksums, an isomorphism proof, and a
  reprofile because bottlenecks shift.
- Alien Artifact Family 34: eigendecomposition and QR-family matrix methods
  require factorization accuracy, orthogonality/residual checks, conditioning
  notes, and explicit numerical stability budgets.

Local negative evidence to respect:

- qglh3 rejected values-only AED suffix deflation: initial `1.066x` did not
  survive final same-worker confirmation.
- qglh3 rejected full-window all-or-nothing AED: it preserved goldens but
  regressed full `eig`; deflating the whole 16x16 window was too rare and too
  expensive.
- Prior notes already rule out q_acc replay, back-substitution-only work, and
  threshold-only AED tuning.

## Candidate Matrix

| Candidate | Impact | Confidence | Effort | Score | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Private AED window-record extractor with residual/orthogonality/shift-list proof, no public dispatch | 3 | 4 | 2 | 6.0 | Select for Pass 3 |
| Direct small-bulge multishift QR public path | 5 | 2 | 5 | 2.0 | Too much for first source lever |
| Partial AED with Schur-block reordering and q_acc window transform | 5 | 2 | 4 | 2.5 | Needs record proof first |
| More threshold-only AED variants | 1 | 1 | 2 | 0.5 | Reject |
| q_acc/eigenvector-only replay changes | 2 | 1 | 3 | 0.67 | Reject |

## Selected Pass 3 Lever

Implement a private, test-only AED window-record helper. It must not change
public `eig`/`eigvals` dispatch in Pass 3.

Required record fields:

- `kw`: top of the trailing window;
- `en`: active block bottom;
- `nw`: actual window size;
- copied Hessenberg window before Schur factoring;
- Schur values from the existing `eig_francis_schur`;
- Schur-vector matrix `Z` for the window;
- conservative deflation count;
- undeflated shift list in current interleaved `(re, im)` ordering.

Acceptance for Pass 3:

- focused unit proof that `W * Z ~= Z * T`;
- `Z^T Z ~= I`;
- shift ordering preserves conjugate adjacency and the current interleaved
  convention;
- no public QR/eig dispatch change;
- no RNG;
- `eigvals_golden` output remains unchanged on the available golden path;
- source remains private/test-only unless a later same-worker rebench moves
  `eig` or `eigvals` above Score `>= 2.0`.

This is a scaffold pass, but it is not ornamental: it is the minimum artifact
needed to avoid repeating the failed threshold/AED variants and to unlock either
partial AED or a measured multishift sweep in the next source pass.
