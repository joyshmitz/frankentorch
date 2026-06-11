# frankentorch-qglh3 pass 5 AED window-record contract

Date: 2026-06-11

## Why This Primitive

The values-only AED suffix shortcut failed final same-worker confirmation. The
next qglh3 swing must be structurally different: a reusable AED window record
that can support either full q_acc back-transform or the `frankentorch-npxbw`
small-bulge multishift sweep.

This pass is contract-only. No public dispatch should change until the record has
focused proof and a same-worker speedup path.

## Record Shape

The next source pass should introduce a private/test-only record with:

- `kw`: top row/column of the trailing AED window;
- `en`: active block bottom;
- `nw`: fixed window size, initially 16 or 32;
- copied Hessenberg window before transformation;
- Schur values from `eig_francis_schur`;
- Schur vectors `Z`;
- conservative deflation count;
- undeflated shifts in current public ordering.

The record must be allocated outside the inner bulge loop where possible. It must
not create unbounded heap churn.

## Proof Obligations

Before public full-`eig` wiring:

- window residual: `W * Z ~= Z * T` within qgce4 dense-eig tolerance;
- orthogonality: `Z^T * Z ~= I`;
- Hessenberg compatibility: either the spike is deflated/restored or the global
  matrix is left unchanged;
- ordering: shifts preserve the current interleaved `(re, im)` convention and
  conjugate adjacency;
- RNG: none;
- strict golden: n64/n128/n256 `eigvals_golden` digests remain unchanged until a
  tolerance-parity public dispatch is explicitly approved;
- focused tests: `eigvals_matches_eig`, complex companion roots, and at least one
  constructed clustered window.

## Acceptance

Keep only if the next source pass scores `>= 2.0`.

Valid keep paths:

- full AED/q_acc transform moves `eig_f64_256x256` on a clean same-worker run;
- the record produces a deterministic shift list that unlocks a measured
  multishift sweep in `frankentorch-npxbw`;
- zero public behavior change plus strong test-only proof is acceptable only as
  a scaffold if immediately followed by the measured multishift lever.

Invalid paths:

- more values-only threshold tuning;
- relaxed deflation without a spike-vector/window residual proof;
- full `eig` output changes without reconstruction/orthogonality evidence.
