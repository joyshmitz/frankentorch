# frankentorch-kgs4.140 BatchNorm1d Scalar-Loss Algebraic-Zero Gauntlet

Date: 2026-06-20
Agent: cod-b / IvoryDeer

## Lever

Specialized f64 training BatchNorm scalar-loss backward:

- `dx = 0`
- `dweight = 0`
- `dbias = upstream * batch * spatial`

For training statistics, the centered normalized terms sum to zero per channel,
so `sum(batch_norm(x, weight, bias))` is independent of `x` and `weight` apart
from tiny dense-reference floating residue. The product path now returns the
annihilated gradients directly rather than rereading the input and emulating a
dense all-ones upstream buffer.

## Benchmark

Command family:

```bash
AGENT_NAME=cod-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
  rch exec -- cargo bench -p ft-api --bench ops_bench --profile release -- \
  batch_norm/grad_1d_ncl_16x128x256 --sample-size 10 --warm-up-time 1 \
  --measurement-time 3 --noplot
```

Workload: f64 BatchNorm1d training scalar backward, `[N,C,L]=[16,128,256]`,
affine weight and bias require gradients.

## Results

| Row | Worker | Native | Scalar Sum | Fold Reference |
|---|---:|---:|---:|---:|
| Initial unpatched baseline | `hz2` | `6.4707 ms` | `3.8543 ms` | `46.121 ms` |
| Patched support run | `vmi1152480` | `5.3185 ms` | `4.1376 ms` | `54.591 ms` |
| Same-worker unpatched retake | `vmi1152480` | `5.6853 ms` | `5.8463 ms` | `56.777 ms` |
| Same-worker patched confirmation | `vmi1152480` | `4.6475 ms` | `4.1630 ms` | `54.596 ms` |

Same-worker ratios from unpatched retake to patched confirmation:

- Native: `1.2233x` faster.
- Scalar sum: `1.4043x` faster.
- Fold reference: `1.0399x` faster.

Criterion final confirmation reported:

- Native change: `[-34.983% -25.382% -14.510%]`, `p = 0.00`.
- Scalar-sum change: `[-37.349% -30.188% -20.954%]`, `p = 0.00`.
- Fold-reference change: `[-23.434% -13.524% -1.5604%]`, `p = 0.04`.

## PyTorch Comparator

Local PyTorch:

- `torch_version=2.12.1+cpu`
- threads/inter-op: `32/32`
- median: `0.956812 ms`
- mean: `1.129408 ms`
- min: `0.780639 ms`
- p95: `2.230037 ms`

Ratios after the kept patch:

- Native/PyTorch: `4.857x` slower.
- Scalar-sum/PyTorch: `4.351x` slower.
- Win/loss/neutral vs PyTorch: `0W / 1L / 0N`.

## Correctness Notes

The first full BatchNorm kernel filter failed the old exact-bit dense-reference
test because the dense formula returns tiny nonzero floating residue for
algebraically zero `dx` and `dweight`. A local PyTorch residue probe found:

- Spatial `1`: max absolute `dx` residue `9.469693939924459e-17`, max
  `dweight` residue `8.579287036987381e-17`, `dbias = [4,4,4]`.
- Spatial `3`: max absolute `dx` residue `3.0141251449398127e-16`, max
  `dweight` residue `2.7829414683458537e-15`, `dbias = [12,12,12]`.

The test now asserts exact product zero for `dx` and `dweight`, bounds the
dense-reference residue, and keeps `dbias` bit-exact.

## Gates

- `rch exec -- cargo test -p ft-api functional_batch_norm1d_tensor_sum --lib --profile release -- --nocapture`: passed, `2 passed`.
- `rch exec -- cargo test -p ft-kernel-cpu batch_norm --lib --profile release -- --nocapture`: first run failed the old exact-bit scalar-backward test; retained as negative evidence.
- `rch exec -- cargo test -p ft-kernel-cpu batch_norm --lib --profile release -- --nocapture`: passed after test-contract update, `7 passed`.
- `rch exec -- cargo test -p ft-conformance --profile release`: passed, conformance green.
- After a manual touched-hunk style fix, `rch exec -- cargo test -p ft-kernel-cpu batch_norm --lib --profile release -- --nocapture`: passed again, `7 passed`.
- Final `rch exec -- cargo check -p ft-kernel-cpu --lib`: passed.
- Final `rch exec -- cargo clippy -p ft-kernel-cpu --lib -- -D warnings`: passed.
- `rustfmt --edition 2024 --check crates/ft-kernel-cpu/src/lib.rs`: still fails on pre-existing whole-file drift outside this lane; the touched BatchNorm assertion hunk was manually formatted and no longer appears in the after-manual-format rustfmt diff.
- `git diff --check`: passed.
- `ubs crates/ft-kernel-cpu/src/lib.rs docs/NEGATIVE_EVIDENCE.md docs/RELEASE_READINESS_SCORECARD.md artifacts/perf/frankentorch-kgs4.140/gauntlet_20260620T1908Z/summary.md`: completed with `0` critical issues and the existing large-file warning inventory.
- After rebasing on `origin/main`, where a same-bead saved-`rstd` scalar-backward path had landed, the final source keeps the algebraic-zero path and re-ran: `ft-kernel-cpu` BatchNorm tests 7/0, `ft-api` tensor-sum tests 2/0, full `ft-conformance` green, and `ft-kernel-cpu` clippy green.

## Verdict

Keep the lever. It is a measured same-worker internal win and narrows the
BatchNorm1d scalar-loss gap. It does not dominate PyTorch. Do not retry this
same scalar-backward algebra; the next gap is forward output materialization,
saved-stat/workspace reuse, tape/session allocation, generated fused code, or
f64-native layout.
