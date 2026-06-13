# frankentorch-gxpb2 Pass 10 Route Closeout

Date: 2026-06-13T16:37:00Z
Agent: `IvoryDeer`
Bead: `frankentorch-gxpb2`

## Scope

This pass finished the `gxpb2` follow-on lane as a rejected/rerouted parent
dispatch bead. No production source was edited.

The clean detached worktree was used because the shared checkout already had
peer-owned `crates/ft-kernel-cpu/src/lib.rs` edits for a different perf bead.

## Baseline

Fresh crate-scoped RCH Criterion baseline on `vmi1227854`:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- \
  cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench \
  eigvals_f64_256x256 -- --warm-up-time 1 --measurement-time 3 --sample-size 10
```

Result:

```text
eigvals_f64_256x256     time:   [24.557 ms 25.483 ms 26.442 ms]
```

The affected large-size rows remain the pass-9 same-worker profile on
`vmi1227854`:

```text
n=512   eigvals=375.76ms  sweeps=583
n=1024  eigvals=2939.18ms sweeps=1132
fallback=0 exceptional=0
```

A pass-10 rerun of `eig_timing_probe` was attempted with remote-only semantics,
but RCH selected degraded `vmi1149989` and correctly refused local fallback. The
pass-9 large-size profile is therefore the decisive route evidence for the
large-n surface, while the pass-10 Criterion row confirms the current n256
strict fallback baseline.

## Candidate Considered

The helper-agent pass recommended a values-only `n >= 512` whole-window AED gate:
copy a trailing window, Schur-factor it, and deflate the window when a boundary
spike test passes.

That candidate was rejected before source edit because it repeats the already
failed qglh3/gxpb2 AED threshold family:

- qglh3 pass 3 rejected values-only AED suffix deflation after the final
  same-worker retry regressed.
- qglh3 pass 4 rejected whole-window threshold AED with q_acc as too rare and
  too expensive.
- gxpb2 pass 8 rejected a copied AED-window record when linked into the public
  build.

Score: `0.0`. The pass did not introduce a source hunk.

## Isomorphism Proof

Because no production source changed in pass 10:

- eigenvalue ordering and complex-pair slot order are unchanged;
- selected-`m`, shift samples, deflation thresholds, and fallback behavior are
  unchanged;
- floating-point arithmetic and RNG behavior are unchanged;
- strict `eigvals_golden` remains the current fallback contract:
  `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.

The decisive behavior proof is the absence of any retained source diff.

## Reroute

`gxpb2` has served as a parent/dispatch integration lane and now has ten passes
of evidence. The remaining work is not another `gxpb2` sub-pass; it belongs to
the deeper `frankentorch-fql10` lane already claimed by `BlackThrush`:

- partial AED with Schur-block reordering and explicit undeflated shifts, or
- scalar-complete small-bulge multishift machinery with BLAS-3 far updates.

The next implementation must reduce sweep count on n512/n1024 and preserve the
n64/n128/n256 strict fallback SHA. Further scalar branch/range/packing edits,
threshold-only AED variants, proof-only records, and public dispatch wiring
before strict proof are rejected.

## Closeout Decision

Close `frankentorch-gxpb2` as rejected/rerouted to `frankentorch-fql10`; do not
leave a duplicate in-progress parent bead competing with the active fql10 owner.
