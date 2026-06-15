# frankentorch-fql10 pass 3: Hessenberg panel scratch reuse rejection

Date: 2026-06-15
Agent: IvoryDeer
Worker: vmi1227854

## Target

`frankentorch-fql10` remains the general non-symmetric eigensolver perf bead.
The profile-backed public rows are `eig_f64_256x256` and
`eigvals_f64_256x256`.

Fresh baseline before any source edit:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- \
  cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- \
  'eigvals_f64_256x256|eig_f64_256x256' \
  --warm-up-time 1 --measurement-time 3 --sample-size 10
```

```text
eig_f64_256x256         [53.090 ms 56.346 ms 58.649 ms]
eigvals_f64_256x256     [22.521 ms 23.155 ms 23.920 ms]
```

## Lever Tried

Hoist the temporary `s`, `col`, `pvec`, `qlin`, and `tcol` buffers in
`hessenberg_reduce_blocked` from reflector scope to panel scope. This was an
allocation/memory-churn lever, not another Francis QR row/column scheduling
micro-cut.

The hunk preserved arithmetic order, shift selection, eigenvalue ordering,
tie-breaking, and RNG behavior; it only changed heap allocation shape.

## Behavior Proof While Candidate Was Present

Remote crate check passed:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- \
  cargo check -j 1 -p ft-kernel-cpu --lib --benches
```

Focused eig/eigh tests passed:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- \
  cargo test -j 1 -p ft-kernel-cpu --lib eig -- --nocapture

26 passed; 0 failed; 445 filtered out
```

Strict golden digests stayed unchanged:

```text
n=64  eigvals=0xbc0583d464b1a211 eig=0xbc0583d464b1a211
n=128 eigvals=0x763c4b15d92c4b89 eig=0x763c4b15d92c4b89
n=256 eigvals=0x00b87b4996340204 eig=0x00b87b4996340204
```

UBS on `crates/ft-kernel-cpu/src/lib.rs` reported 0 critical issues. It still
reports the existing large warning inventory in this crate.

## Rebench

Unconditional scratch reuse:

```text
eig_f64_256x256         [47.771 ms 49.804 ms 52.057 ms]
eigvals_f64_256x256     [24.173 ms 24.437 ms 24.934 ms]
```

The full `eig` row improved, but `eigvals` regressed against the same-worker
baseline (`23.155 ms -> 24.437 ms`). This fails the shared QR-floor keep gate.

Narrowed full-eig-only scratch reuse (`accumulate_q` gated):

```text
eig_f64_256x256         [50.143 ms 53.120 ms 56.589 ms]
eigvals_f64_256x256     [23.405 ms 24.232 ms 25.134 ms]
```

The full `eig` interval overlapped baseline and `eigvals` still regressed
(`23.155 ms -> 24.232 ms`).

## Verdict

Rejected. Score `0.0`: one target row regressed and the full `eig` improvement
did not remain decisive after narrowing.

The source hunk was removed; `crates/ft-kernel-cpu/src/lib.rs` has no retained
diff for this pass.

## Route

Do not repeat Hessenberg scratch-shape changes for fql10. The next pass should
return to the deeper primitive already selected by prior evidence:

- partial AED with Schur-block reordering and explicit undeflated shifts; or
- a strict-fallback Schur-window kernel / small-bulge multishift path that
  reduces sweep count rather than allocation shape.
