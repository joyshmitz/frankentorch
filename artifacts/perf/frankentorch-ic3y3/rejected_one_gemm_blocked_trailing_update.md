# frankentorch-ic3y3 - one-GEMM blocked trailing update rejected

## Target

- Bead: `frankentorch-ic3y3`
- Surface: `ft-kernel-cpu` symmetric eigensolver reduction
- Public baseline row: `eigh_f64_256x256` / `eigvalsh_f64_256x256`
- Primitive tested: replace the full-matrix blocked reducer's two-GEMM trailing update with the existing finite one-GEMM lower rank-2k update plus upper-triangle mirror.

## Fresh Baseline

Focused RCH Criterion selected `vmi1149989`:

```text
eigh_f64_256x256                        [11.392 ms 11.757 ms 12.135 ms]
eigvalsh_f64_256x256                    [5.4395 ms 5.5893 ms 5.8520 ms]
sym_rank2k_lower_scalar_f64_256x32      [611.43 us 637.44 us 673.65 us]
sym_rank2k_lower_gemm_f64_256x32        [161.18 us 163.26 us 167.18 us]
```

Evidence:

- `artifacts/perf/frankentorch-dsytrd-reprofile-20260615/baseline_linalg_vmi1227854.log`

Note: `RCH_WORKERS=vmi1227854` was requested, but RCH selected `vmi1149989`; all `ic3y3` AB evidence below is therefore compared on `vmi1149989`.

## Existing Blocked AB

Before editing, the existing full-matrix blocked reducer was still not an admissible public dispatch route:

```text
n=256   packed=   4.899ms blocked=   5.902ms  ratio=0.83x  maxdiff=9.09e-13
n=512   packed=  39.105ms blocked=  37.498ms  ratio=1.04x  maxdiff=2.27e-12
n=768   packed= 120.522ms blocked= 141.228ms  ratio=0.85x  maxdiff=7.84e-12
n=1024  packed= 342.113ms blocked= 359.373ms  ratio=0.95x  maxdiff=1.34e-11
```

Evidence: `pass1_existing_blocked_ab_vmi1149989.log`.

## Candidate

One lever:

- In `eigh_tridiag_reduce_blocked`, replace the trailing-block pair of GEMMs:
  - `V W^T`
  - `W V^T`
- with:
  - copy the trailing `A0[pe..n, pe..n]` block;
  - call `symmetric_rank2k_lower_update_f64(mt, nb, V, W, trailing)`;
  - mirror lower triangle to upper;
  - copy the full trailing block back.

The intended contract was finite-input bit-equivalence to the two-GEMM form for the blocked experimental reducer, plus lower memory traffic in the trailing update.

## Candidate Result

Same worker family and same `eigvalsh_blocked_ab` harness:

```text
n=256   packed=  15.294ms blocked=  23.378ms  ratio=0.65x  maxdiff=9.66e-13
n=512   packed=  88.415ms blocked=  79.819ms  ratio=1.11x  maxdiff=1.99e-12
n=768   packed= 255.303ms blocked= 271.189ms  ratio=0.94x  maxdiff=1.30e-11
n=1024  packed= 541.507ms blocked= 818.830ms  ratio=0.66x  maxdiff=1.25e-11
```

Evidence: `pass3_candidate_blocked_ab_vmi1149989.log`.

## Verdict

REJECT.

- The target public 256x256 lane remains slower (`0.65x` in the candidate AB, after an already-slower `0.83x` blocked baseline).
- Larger sizes are mixed and do not clear the Score gate.
- The source hunk was removed; no runtime source change is retained.
- Score: `0.0`.

## Proof Notes

- The candidate compiled.
- The first focused test command used a literal `|` Cargo filter and ran zero tests; no passing proof is claimed from that log.
- The AB harness still produced tolerance-scale eigenvalue differences (`~1e-12`), but the performance gate failed before broader proof gates were useful.

## Next Route

Do not retry full-matrix WY trailing-update rewrites or public dispatch through `eigvalsh_blocked_f64`.

The next admissible no-gaps primitive must change the memory-traffic model, not just the trailing update kernel:

- band-packed compact-WY panel reduction that avoids full `n x n` trailing-block traffic; or
- a tridiagonal divide-and-conquer / secular solver for the post-reduction full-`eigh` residual, after a fresh split profile proves the QL/vector path is the dominant remaining wall.
