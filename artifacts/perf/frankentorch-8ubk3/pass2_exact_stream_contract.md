# frankentorch-8ubk3 Pass 2 Exact-Stream Proof Contract

Date: 2026-06-12

Scope: proof-contract artifact only for `frankentorch-8ubk3`.
No production source was edited in this pass.

## Baseline Inputs

Primary baseline/profile artifact:

- `artifacts/perf/frankentorch-8ubk3/pass1_baseline_profile.md`

Strict golden SHA-256:

```text
24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725
```

Profile-backed residual:

```text
hz1  eigvals_f64_256x256 [33.839 ms 34.014 ms 34.212 ms]
hz1  eig_f64_256x256     [68.327 ms 68.927 ms 69.570 ms]
ovh-a n=256  sweeps=319  defl1=14 defl2=121 fallback=0 exceptional=0
ovh-a n=1024 sweeps=1132 defl1=18 defl2=503 fallback=0 exceptional=0
```

Prior rejection:

- `frankentorch-fy8to` pass 4 changed the n=256 values-only digest by changing
  shift policy, so `frankentorch-8ubk3` must preserve the shift stream.

## Required Isomorphism

Any source candidate must prove these streams are identical to the scalar
baseline for the same fixture before a benchmark result can be considered:

1. `EigFrancisShiftSample` stream:
   - `active_first`
   - `active_last`
   - `iteration_in_window`
   - `accumulated_shift`
   - `x`
   - `y`
   - `w`
   - `exceptional`
2. Selected bulge start stream:
   - every recorded `selected_m`
   - same length and same position-by-position value
3. Deflation counters:
   - 1x1 deflations
   - 2x2 deflations
   - fallback deflations
   - max-total exhaustions
   - exceptional shifts
4. Eigenvalue slot writes:
   - bottom-up 1x1 order unchanged
   - 2x2 real-pair upper/lower slot order unchanged
   - complex conjugate pair sign convention unchanged
   - no sorting, tie repair, or signed-zero normalization
5. Global deterministic surface:
   - no RNG
   - no hash iteration order
   - no parallel schedule dependence
   - strict `eigvals_golden` stdout SHA unchanged

## Allowed Source Shape

Allowed next source lever:

```text
private exact-shift sweep proof harness or blocked-sweep source slice
```

The source slice may reorganize execution only after the current scalar shift
source and scalar m-search have produced the same stream. A candidate can
introduce a reflector ledger, tiled active-window scratch, or blocked row/column
application only if it proves the observable stream above is identical.

Public `eig`/`eigvals` dispatch must keep scalar fallback unless the source
slice passes this exact-stream proof plus strict golden.

## Disallowed Source Shape

Reject these immediately:

- alternate shift packets
- AED-derived shift replacement
- direct two-bulge/four-shift shift-policy changes
- range/index micro-cuts
- diagnostic-only helpers without a path to a source speedup
- tolerance-based deflation repairs
- source changes that only help full `eig` while changing values-only behavior

## Floating-Point Ledger

The candidate can be keepable only under one of two ledgers:

1. Bit-exact global update ledger:
   - same per-slot operation order for every value read by subsequent scalar
     recurrence
   - expected to preserve strict digest
2. Strict fallback ledger:
   - candidate declines and scalar path runs
   - behavior is preserved, but performance score is zero unless a separate
     candidate path actually executes on the profiled row

If tiled/blocked application reassociates operations that feed future shifts or
deflations, it is rejected unless strict golden and the recorded streams still
match exactly.

## Validation Gates

Minimum gates for the next source pass:

```text
rch exec -- cargo test -p ft-kernel-cpu --lib eig -- --nocapture
rch exec -- cargo run --release -q -p ft-kernel-cpu --example eigvals_golden
rch exec -- cargo check -p ft-kernel-cpu --lib --examples --benches
rch exec -- cargo clippy -p ft-kernel-cpu --lib --examples --benches -- -D warnings
rch exec -- cargo fmt -p ft-kernel-cpu --check
ubs crates/ft-kernel-cpu/src/lib.rs
```

The exact-stream comparison can be implemented as a private focused test or
diagnostic example, but it must be committed as part of the source pass if the
candidate is kept.

## Pre-Score

Candidate family:

```text
strict scalar-shift-preserving blocked Francis sweep machinery
```

Score:

```text
Impact 4: the 319/1132-sweep stream is still the measured floor.
Confidence 3: preserving shift and m streams avoids the fy8to digest failure.
Effort 4: proof harness plus one source slice and same-worker rebench.
Score = 4 * 3 / 4 = 3.0
```

## Stop Rules

Remove the source hunk and record rejection if any of these occur:

- strict stdout SHA changes
- shift sample stream differs
- selected `m` stream differs
- deflation counters differ
- candidate always falls back
- same-worker median fails Score >= 2.0

## Verdict

Productive proof-contract pass. Pass 3 should refine the exact-shift blocked
sweep primitive against the communication-avoiding Hessenberg QR route, then
Pass 4 can attempt exactly one source slice if the score still clears the gate.
