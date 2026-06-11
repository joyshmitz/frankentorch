# Pass 6 - Two-Bulge Source Pilot Rejection

Date: 2026-06-11
Bead: `frankentorch-npxbw`
Worker target: `vmi1227854`

## Decision

Rejected source implementation for this pass. `crates/ft-kernel-cpu/src/lib.rs`
was left unchanged.

The pass-5 route is algorithmically correct: the remaining non-symmetric
`eigvals` wall is the scalar Francis QR sweep, and the right replacement family
is small-bulge multishift QR. The requested first source slice, however, is not
a narrow safe hunk in the current implementation. The active code is an
EISPACK-style in-place double-shift sweep:

- one double-shift source `(x, y, w)` is selected from the trailing 2x2 block;
- `m` is selected by the existing two-subdiagonal test;
- a single 3-row Householder bulge is chased from `m` through `na`;
- row updates and column updates are interleaved immediately in scalar order;
- deflation, exceptional shifts, and `max_total` fallback are observed after each
  scalar sweep.

A real two-bulge/four-shift pass would change the transform sequence and the
interleaving of row/column updates even before any far-update batching. That is
not bit-equivalent to the scalar fallback, and the pass gate requires the strict
`eigvals_golden` stdout SHA:

```text
24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725
```

Implementing the pilot correctly would require more than adding a shift packet:

1. define the four-shift polynomial and two starting bulges without changing the
   current `m` search semantics;
2. handle bulge spacing, collision, and active-window ends for every deflation
   boundary;
3. preserve the Hessenberg invariant after every local packet update;
4. preserve the existing exceptional-shift cadence and fallback accounting;
5. prove the strict digest on the real production path, not on a diagnostic-only
   shadow path.

The only source hunk small enough for one pass would be diagnostic-only shift
packet scaffolding, but pass 3 already added the shift/profile scaffold. Adding
another unused helper would not implement the requested lever, would not reduce
sweeps, and would not clear the Score gate.

## Proof Run On Current Source

Focused eig tests:

```bash
RCH_WORKER=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
  rch exec -- cargo test -p ft-kernel-cpu --lib eig -- --nocapture \
  > artifacts/perf/frankentorch-npxbw/pass6_reject_current_test_eig.log 2>&1
```

Result: PASS on remote `vmi1227854`; `21 passed; 0 failed`.

Strict golden:

```bash
RCH_WORKER=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
  rch exec -- cargo run --release -q -p ft-kernel-cpu --example eigvals_golden \
  > artifacts/perf/frankentorch-npxbw/pass6_reject_current_eigvals_golden.strict.stdout \
  2> artifacts/perf/frankentorch-npxbw/pass6_reject_current_eigvals_golden.log
```

RCH emitted the program stdout into its log stream, so the nine digest lines were
extracted to:

```text
artifacts/perf/frankentorch-npxbw/pass6_reject_current_eigvals_golden.extracted.strict.stdout
```

Hash:

```text
24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725  artifacts/perf/frankentorch-npxbw/pass6_reject_current_eigvals_golden.extracted.strict.stdout
```

Crate check:

```bash
RCH_WORKER=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
  rch exec -- cargo check -p ft-kernel-cpu --lib --examples --benches \
  > artifacts/perf/frankentorch-npxbw/pass6_reject_current_check_ft_kernel_cpu.log 2>&1
```

Result: PASS on remote `vmi1227854`.

Format:

```bash
RCH_WORKER=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
  rch exec -- cargo fmt -p ft-kernel-cpu --check \
  > artifacts/perf/frankentorch-npxbw/pass6_reject_current_fmt_check.log 2>&1
```

Result: RCH refused because `cargo fmt` is a non-compilation command while
remote fallback was required. The exact scoped fmt check was then run locally:

```bash
CARGO_TERM_COLOR=never cargo fmt -p ft-kernel-cpu --check \
  > artifacts/perf/frankentorch-npxbw/pass6_reject_current_fmt_check_local.log 2>&1
```

Result: PASS.

## Benchmark Gate

No candidate source was benchmarked because no source lever was kept long enough
to test. The required current comparison row remains the pass-5/pass-3 gate:

```text
eigvals_f64_256x256 on vmi1227854: [25.052 ms 25.370 ms 25.694 ms]
```

No after row was generated.

Supplemental current-source bench attempted during closeout:

```bash
RCH_WORKER=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
  rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench eigvals_f64_256x256 \
  > artifacts/perf/frankentorch-npxbw/pass6_baseline_eigvals_f64_256x256_vmi1227854.log 2>&1
```

RCH selected `vmi1153651`, not `vmi1227854`, so this row is non-decisive for
same-worker keep/reject scoring. It measured current source only:

```text
eigvals_f64_256x256 on vmi1153651: [52.616 ms 54.271 ms 56.297 ms]
```

## Route

Do not repeat range/index micro-cuts and do not add another inert shift helper.
The next source pass should start from an AED-derived shift-list or a
standalone private Schur-window kernel that can be verified against strict
fallback before it is wired into `eig_francis_schur`. A production two-bulge
replacement should only land once it owns deflation-boundary handling and has a
separate strict fallback gate for unsupported active windows.

## Verdict

REJECTED / SOURCE UNCHANGED.
