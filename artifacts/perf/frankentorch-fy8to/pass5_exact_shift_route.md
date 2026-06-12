# frankentorch-fy8to Pass 5 Exact-Shift Deeper Route

Date: 2026-06-12

Scope: post-rejection route artifact for `frankentorch-fy8to`; no production
source was edited in this pass.

## Diagnosis

Pass 4 proved that an AED-derived alternate shift packet can pass focused eig
tests while still changing the strict `eigvals` bit digest for the profile and
golden fixture. Under this bead's strict contract, a shift-policy change is not
the next keepable primitive.

The measured target remains the same non-symmetric Francis QR floor:

```text
vmi1227854 n=256  sweeps=319  fallback=0  exceptional=0
vmi1227854 n=1024 sweeps=1132 fallback=0  exceptional=0
```

The next algorithmic lever must preserve, not replace, the current scalar
sequence:

- same active-window observations
- same `x/y/w` shift samples
- same selected `m` stream
- same 1x1 and 2x2 deflation order
- same complex-pair slot convention
- same strict stdout SHA-256

## Primitive To Attack Next

Build a strict scalar-shift-preserving blocked Francis sweep kernel.

The primitive target is not a new shift policy. It is a different execution
model for the same sequence of implicit double-shift transformations:

1. Keep the current scalar shift source and m-search as the truth source.
2. Record a per-sweep reflector ledger for the existing single-bulge chase.
3. Apply only those reflector operations whose dependencies are proven local,
   in tiled row/column panels, preserving per-slot arithmetic order.
4. Fall back to the current scalar loop if the ledger cannot prove identical
   read/write order for a sweep.

The first source slice should be a private proof harness, not public dispatch:
compare current scalar `EigFrancisShiftSample`, selected `m`, deflation counts,
and strict `eigvals_golden` digest against the blocked-sweep candidate before
running a same-worker performance gate.

## Pre-Score

```text
Impact 4: the 319/1132-sweep floor is still the geev/eigvals wall.
Confidence 3: preserving the shift stream removes the pass-4 digest failure mode,
              but exact H update order is still a hard proof obligation.
Effort 4: private ledger/proof harness plus focused gates and same-worker rebench.
Score = 4 * 3 / 4 = 3.0
```

## Stop Rules

Reject the next source slice immediately if any of these occur:

- strict golden SHA changes
- shift sample stream differs
- selected `m` stream differs
- deflation counters differ
- any candidate path only falls back and therefore has no performance effect
- same-worker median does not clear Score >= 2.0

This route supersedes `frankentorch-fy8to`'s alternate-shift-list attempt.
