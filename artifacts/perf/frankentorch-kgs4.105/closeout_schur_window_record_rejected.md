# frankentorch-kgs4.105 closeout: standalone Schur-window record rejected

## Target

Profile-backed route from the closed fql10/qglh3/npxbw chain: `eigvals_f64_256x256`
remains on the scalar Francis QR floor, and prior AED/threshold/two-bulge source
pilots did not clear the keep bar. This pass tried one diagnostic lever: a hidden
standalone Schur-window record path for the next multishift/AED implementation.

## Local baseline

Command:

```text
env -u RCH_REQUIRE_REMOTE CARGO_TARGET_DIR=/data/tmp/frankentorch-a7xya-local-target \
  cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- \
  '^(eigvals_f64_256x256|eig_f64_256x256)$' \
  --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Evidence: `pass1_local_baseline_eig_eigvals_256.log`

- `eig_f64_256x256`: `[44.543 ms 45.415 ms 46.655 ms]`
- `eigvals_f64_256x256`: `[25.963 ms 26.795 ms 27.710 ms]`

## Candidate proof

Candidate source added a private trace sink that copied a bounded trailing
Hessenberg window, Schur-factorized it standalone, and recorded:

- `kw/en/nw`
- copied window `W`
- standalone Schur form `T`
- Schur vectors `Z`
- interleaved Schur values and undeflated shift list
- conservative deflation count `0`
- `W*Z - Z*T` residual
- `Z^T*Z - I` residual
- conjugate adjacency check

Focused proof command:

```text
env -u RCH_REQUIRE_REMOTE CARGO_TARGET_DIR=/data/tmp/frankentorch-a7xya-local-target \
  cargo test -j 1 -p ft-kernel-cpu eig_francis_ -- --nocapture
```

Evidence: `pass1_window_record_focused_tests.log`

- 5 focused eig diagnostic tests passed.
- Standalone window proof residuals met the candidate thresholds:
  `W*Z-Z*T < 1e-6`, `Z^T*Z-I < 1e-10`.
- Shift-list FNV digest while candidate was present:
  `0x299f1fb3334e65ec`.
- Production eig/eigvals output was not routed through the diagnostic path.

Strict golden command:

```text
env -u RCH_REQUIRE_REMOTE CARGO_TARGET_DIR=/data/tmp/frankentorch-a7xya-local-target \
  cargo run -j 1 -p ft-kernel-cpu --release --example eigvals_golden
```

Evidence:

- `pass1_eigvals_golden.log`
- `pass1_eigvals_golden.sha256`

Golden SHA-256:

```text
38aeb596e0e918773347d166af30dacab717d93d4bfa210e815be48e92bdbbb9
```

## Rebench

Evidence: `pass1_local_rebench_eig_eigvals_256.log`

- `eig_f64_256x256`: `[46.107 ms 46.644 ms 47.336 ms]`
  - Criterion change: `+3.9120%`, `p = 0.01`, reported regression.
- `eigvals_f64_256x256`: `[25.634 ms 26.750 ms 27.785 ms]`
  - Criterion change: `+1.8272%`, `p = 0.57`, no significant change.

## Decision

Rejected. Score `0.0`: the candidate was diagnostic-only, public rows did not
improve, and `eig_f64_256x256` regressed in the local Criterion comparison.

Source/test hunks were removed after the rejection; `crates/ft-kernel-cpu/src/lib.rs`
has no remaining diff from this pass.

## Reroute

Do not repeat Schur-window scaffolding as a standalone commit. The next
profile-backed route should be a production algorithmic lever with direct public
row impact, most likely:

- integrate an AED-style deflation test into the live active Francis window, or
- replace the scalar double-shift sweep with a bounded multishift bulge-chase
  path that consumes a prevalidated shift list and proves eig/eigvals golden
  output unchanged.

Keep the one-lever rule: baseline, source lever, strict output proof, rebench,
and reject unless Score is at least `2.0`.
