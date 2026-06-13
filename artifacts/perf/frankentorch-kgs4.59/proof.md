# frankentorch-kgs4.66 proof bundle

Note: this artifact directory was created under the original scratch bead ID
`frankentorch-kgs4.59`. During rebase, `origin/main` already contained a
different closed `frankentorch-kgs4.59`, so the landing bead for this
optimization is `frankentorch-kgs4.66`.

## Profile-backed target

- Source profile note: `artifacts/perf/notes/bandwidth_bound_ops_2026-06-13.md`
- Target: `ft-api` no-grad `tensor_amax`/`tensor_amin` f64 value-only reductions on contiguous last-dimension lanes.
- Chosen lever: safe-Rust f64x4 last-dimension value reduction plus Rayon row parallelism behind the existing no-grad, F64-only API path.

## Benchmark evidence

Same-worker baseline and after are from `vmi1149989`.

| Case | Old median | New median | Speedup |
| --- | ---: | ---: | ---: |
| `amax_amin/amax_f64_lastdim/2048x2048` | 2.6516 ms | 693.59 us | 3.82x |
| `amax_amin/amin_f64_lastdim/2048x2048` | 2.8861 ms | 666.77 us | 4.33x |

Evidence files:

- `baseline_vmi1149989.txt`
- `after_rebased_criterion_vmi1149989.txt`

Score: Impact 5 x Confidence 4 / Effort 3 = 6.67, keep.

## Isomorphism proof

- Ordering preserved: output lanes are written in the same row-major order as the scalar double loop.
- Tie-breaking unchanged: strict `>` / `<` replacement is preserved; equal values keep the earlier value. Signed-zero equality is repaired after SIMD reduction so the first zero bit pattern is returned.
- Floating point unchanged: no reassociation of arithmetic occurs because this is selection-only, not summation. Non-NaN comparisons use the same strict predicates as the old scalar path.
- NaN payload behavior unchanged: the previous `amax`/`amin` scalar closure replaces on every NaN, so multiple NaNs preserve the later NaN payload. The fast path mirrors that rule.
- RNG unchanged: no RNG use or seed/state changes.
- Autograd unchanged: the fast path is gated to `!requires_grad && DType::F64`; grad-enabled inputs continue through the existing tie-splitting backward path.
- DType behavior unchanged for non-F64: all non-F64 inputs continue through the existing cast/restore path.

## Golden output

`golden_output.txt` contains only deterministic `GOLDEN_*` proof lines from the focused kernel/API tests.

Verification:

```text
sha256sum -c artifacts/perf/frankentorch-kgs4.59/golden_output.sha256
artifacts/perf/frankentorch-kgs4.59/golden_output.txt: OK
```
