# frankentorch-qglh3 pass 4 full-window AED rejection

Date: 2026-06-11
Agent: `IvoryDeer`
Bead: `frankentorch-qglh3`

## Target

Pass 3 rejected the values-only threshold suffix. Pass 4 attacked the real
`geev` path instead: full `eig_f64_256x256` with `q_acc` preservation.

Stable comparator from pass 1 on `vmi1227854`:

| Row | Estimate |
| --- | ---: |
| `eig_f64_256x256` | `[49.080 ms 49.892 ms 51.006 ms]` |

Fresh current-head reprofile on the same worker was noisy and slower:

| Row | Estimate |
| --- | ---: |
| `eig_f64_256x256` | `[80.410 ms 102.20 ms 123.52 ms]` |

That row is routing evidence only; the pass-1 row remains the cleaner comparator.

## One Lever Tried

Added a full-window AED candidate that only ran on `want_vectors=true`.
It copied a trailing 16x16 Hessenberg window, recursively Schur-factored it with
`eig_francis_schur`, flushed pending q_acc rotations before applying the window
transform, applied `Z` to `h` and `q_acc`, and deflated the whole window only
when the spike bound was tiny.

This was materially different from pass 3: it targeted full `eig` and respected
the q_acc ordering hazard instead of changing only values-only `eigvals`.

## Proof

- `cargo fmt -p ft-kernel-cpu --check`: passed.
- `cargo test -j 1 -p ft-kernel-cpu eig_ -- --nocapture`: 6 passed.
- `cargo test -j 1 -p ft-kernel-cpu eigvals -- --nocapture`: 5 passed.
- `eigvals_golden` on remote `vmi1227854`: strict n64/n128/n256 digests stayed
  `0xbc0583d464b1a211`, `0x763c4b15d92c4b89`, `0x00b87b4996340204` for both
  eigvals and eig.
- Golden log sha256: see `pass4_full_aed_eigvals_golden.sha256`.

## Rebench

Same-worker candidate row on `vmi1227854`:

| Row | Estimate |
| --- | ---: |
| `eig_f64_256x256` | `[66.837 ms 74.874 ms 86.356 ms]` |

Against the stable pass-1 comparator, this is `49.892 ms -> 74.874 ms`
(`0.666x`). Against the noisy current-head reprofile it is not a credible keep:
the confidence interval is still broad, and the source is more complex while
not clearing the original target.

Score: `0.0`; reject and remove the source hunk.

## Next Route

Do not repeat whole-window threshold AED. The failure mode is that deflating the
entire 16x16 window is too rare/expensive and does not supply reusable shifts.
The next useful primitive is either:

- partial AED with actual Schur block reordering and a shift-list artifact for
  `frankentorch-npxbw`, or
- direct small-bulge multishift QR scaffolding that can consume an explicit shift
  list and batch far updates.

The path forward is still AED/multishift; this pass only rejects the
whole-window all-or-nothing variant.
