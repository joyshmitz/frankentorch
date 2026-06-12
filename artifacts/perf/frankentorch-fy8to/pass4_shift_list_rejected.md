# frankentorch-fy8to Pass 4 Shift-List Source Rejection

Date: 2026-06-12

Scope: one source lever for bead `frankentorch-fy8to`.

## Baseline Before Source

Immediate pre-edit Criterion baseline:

```text
rch worker: hz1
cargo bench -p ft-kernel-cpu --bench linalg_bench -- eigvals_f64_256x256
eigvals_f64_256x256 time: [33.794 ms 34.005 ms 34.240 ms]
```

Historical same-worker anchor from pass 1 remains:

```text
vmi1227854 eigvals_f64_256x256 [24.692 ms 25.049 ms 25.415 ms]
vmi1227854 eig_f64_256x256     [73.003 ms 75.979 ms 79.258 ms]
strict stdout sha256           24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725
```

## Lever Tried

Tried the authorized pass-4 source slice:

```text
AED-derived shift-list consumed by the existing scalar single-bulge chase.
```

Concrete hunk:

- private copied trailing Schur window for `eigvals` only
- wide active-window gate
- copied-window finite and Hessenberg checks
- recursive local Schur planning with recursive planning disabled inside the copy
- scalar m-search validation before packet use
- full `eig` and exceptional-shift paths forced to scalar fallback

The global update schedule remained the current scalar single-bulge chase.

## Candidate Validation

Focused eig/eigvals tests passed on RCH worker `vmi1227854`:

```text
rch exec -- cargo test -p ft-kernel-cpu --lib eig -- --nocapture
21 passed; 0 failed
```

Strict golden rejected the candidate on RCH worker `ovh-a`:

```text
frankentorch-l9xod eigvals_golden n=64
eigvals_digest=0xbc0583d464b1a211
eig_digest=0xbc0583d464b1a211
frankentorch-l9xod eigvals_golden n=128
eigvals_digest=0x763c4b15d92c4b89
eig_digest=0x763c4b15d92c4b89
frankentorch-l9xod eigvals_golden n=256
eigvals_digest=0xdaa6738e0f31a016
eig_digest=0x00b87b4996340204
```

Required `n=256` strict `eigvals_digest`:

```text
0x00b87b4996340204
```

The source hunk changed the values-only convergence path while full `eig`
stayed on scalar fallback. That violates the strict order-sensitive digest gate,
so the lever has no keep path regardless of any potential benchmark result.

## Revert Proof

The source hunk was removed. `crates/ft-kernel-cpu/src/lib.rs` returned to a
zero diff.

Post-revert golden on RCH worker `hz1` restored the required digest:

```text
frankentorch-l9xod eigvals_golden n=64
eigvals_digest=0xbc0583d464b1a211
eig_digest=0xbc0583d464b1a211
frankentorch-l9xod eigvals_golden n=128
eigvals_digest=0x763c4b15d92c4b89
eig_digest=0x763c4b15d92c4b89
frankentorch-l9xod eigvals_golden n=256
eigvals_digest=0x00b87b4996340204
eig_digest=0x00b87b4996340204
```

## Verdict

Rejected.

Score:

```text
0.0
```

Reason: behavior drift in the strict `eigvals` digest.

Next route: do not pursue more alternate-shift packets under this strict digest.
The next primitive must preserve the existing scalar shift/deflation stream and
attack the sweep kernel itself: exact-shift, exact-deflation-order blocked
Francis sweep machinery with a proof that `EigFrancisShiftSample` and selected
`m` streams stay byte-for-byte identical before any benchmark keep decision.
