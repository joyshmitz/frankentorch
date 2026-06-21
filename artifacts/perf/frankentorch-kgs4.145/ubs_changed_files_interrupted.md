# UBS Changed-Files Scan Caveat

Date: 2026-06-21
Agent: cod-a / IvoryDeer

The scoped changed-file UBS scan included the two touched giant Rust source
files, `crates/ft-api/src/lib.rs` and `crates/ft-kernel-cpu/src/lib.rs`. It ran
silently through repeated polls and was interrupted with `Ctrl-C` after the
extra wait interval so the session could land the measured perf evidence. Exit
code was `130`; UBS emitted no findings before interruption.

A follow-up docs/artifact-only invocation:

```text
ubs docs/NEGATIVE_EVIDENCE.md docs/RELEASE_READINESS_SCORECARD.md artifacts/perf/frankentorch-kgs4.145/summary.md
```

exited `0` and reported no recognizable languages in the shadow workspace.
