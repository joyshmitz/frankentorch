# frankentorch-8ubk3 Pass 3 Primitive Refinement

Date: 2026-06-12

Scope: primitive-selection artifact only. No production source was edited.

## Inputs

- Pass 1 baseline/profile: `artifacts/perf/frankentorch-8ubk3/pass1_baseline_profile.md`
- Pass 2 exact-stream contract: `artifacts/perf/frankentorch-8ubk3/pass2_exact_stream_contract.md`
- Prior rejection: `artifacts/perf/frankentorch-fy8to/pass4_shift_list_rejected.md`

Profile-backed floor:

```text
n=256  sweeps=319  fallback=0  exceptional=0
n=1024 sweeps=1132 fallback=0  exceptional=0
```

## Candidate Ranking

### 1. Exact-Shift Blocked Sweep Ledger

Source shape:

```text
current scalar shift source + current selected-m search
        -> per-sweep reflector ledger
        -> dependency-proven tiled row/column application
        -> scalar fallback on any unproven sweep
```

Why it survives pass-2:

- preserves shift policy
- preserves selected `m`
- can compare stream equality before speed gates
- attacks the sweep body instead of the already-rejected shift source

Pre-score:

```text
Impact 4 * Confidence 3 / Effort 4 = 3.0
```

Verdict: selected.

### 2. Full Small-Bulge Multishift / Far-Update WY

This is still the long-range LAPACK-class primitive, but it changes the
transformation schedule before the project has an exact-stream proof harness.
It remains relevant after an exact fallback/proof substrate exists.

Pre-score:

```text
Impact 5 * Confidence 2 / Effort 5 = 2.0
```

Verdict: defer until the exact-shift proof substrate exists.

### 3. Alternate AED Shift Packets

Rejected by `fy8to`: focused eig tests can pass while strict values-only digest
changes. Repeating this family is not acceptable under the current strict
contract.

Score:

```text
0.0
```

Verdict: rejected.

### 4. Range/Index Micro-Cuts

Rejected by previous npxbw/fy8to evidence. They do not attack sweep count and
have already produced same-worker regressions.

Score:

```text
0.0
```

Verdict: rejected.

## Selected Pass-4 Source Slice

Pass 4 may implement exactly one private source slice:

```text
exact-shift sweep ledger proof harness
```

Minimum useful slice:

1. Reuse the existing scalar shift source and selected-`m` search.
2. Record enough per-sweep metadata to compare baseline and candidate streams.
3. Add a focused hidden test or diagnostic path that proves equality of:
   - shift samples
   - selected `m`
   - deflation counters
   - strict golden digest
4. Do not change public dispatch unless this proof passes.

If the proof harness cannot attach to a real blocked-sweep candidate in the
same source family, reject it as diagnostic-only and route to the next deeper
primitive.

## First Real Speed Lever After Harness

Once the proof harness exists, the first source speed lever should target the
strided column modification inside the scalar single-bulge chase:

```text
for i in 0..=jmax {
    update h[i, k], h[i, k+1], h[i, k+2]
}
```

The candidate must prove either:

- exact per-slot operation order is preserved, or
- any reassociation still keeps the strict digest and exact stream equality.

No source keep is allowed from a tolerance-only argument.

## Verdict

Productive primitive-refinement pass. The next pass is allowed to implement one
source slice only if it is the exact-shift sweep ledger/proof substrate for the
blocked sweep kernel, not another shift replacement or range micro-cut.
