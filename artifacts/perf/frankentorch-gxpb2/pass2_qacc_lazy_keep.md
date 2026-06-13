# frankentorch-gxpb2 pass 2 lazy eigvals q_acc keep

Date: 2026-06-13
Bead: `frankentorch-gxpb2`
Worker: `hz1`

## Lever

The public eig/eigvals worker always allocated and initialized an `n*n`
identity `q_acc`, even when `want_vectors == false`. The values-only
Hessenberg and Francis paths already guard every Q update behind
`want_vectors`, and `eigvals_contiguous_f64` discards eigenvectors.

This pass adds `eig_initial_q_acc(n, want_vectors)` and uses an empty vector for
values-only paths. Full `eig` still receives the same identity matrix.

## Isomorphism

- Ordering preserved: yes; Hessenberg and Francis scalar operations are
  unchanged.
- Tie-breaking unchanged: yes; no split, shift, or deflation threshold changed.
- Floating-point unchanged: yes for all live eigenvalue arithmetic; only a dead
  vector sink allocation is skipped for `eigvals`.
- RNG unchanged: no RNG.
- Golden output:
  `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.

## Benchmarks

Same-worker Criterion A/B:

| row | before | after |
| --- | ---: | ---: |
| `eigvals_f64_256x256` | `[51.052 ms 51.500 ms 51.901 ms]` | `[49.954 ms 50.567 ms 51.052 ms]` |

Median speedup: `51.500 / 50.567 = 1.018x`.

The confidence intervals touch at the boundary and the after run had outliers,
so this is recorded as a small, low-risk keep, not as the main `fql10-D`
dispatch win.

## Gates

- `cargo test -j 1 -p ft-kernel-cpu --lib eig -- --nocapture`: passed `24/24`
  on remote `hz1`.
- `eigvals_golden`: strict stdout SHA stayed
  `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.
- `cargo fmt -p ft-kernel-cpu --check`: passed locally.
- `ubs crates/ft-kernel-cpu/src/lib.rs`: exit `0`; no critical issues.
- `cargo check -j 1 -p ft-kernel-cpu --lib --examples --benches`: passed on
  remote `hz1`.
- `cargo clippy -j 1 -p ft-kernel-cpu --lib --examples --benches -- -D warnings`:
  passed on remote `hz1`.

## Score

`Impact 1.018 * Confidence 0.85 / Effort 0.35 = 2.47`.

Keep. `frankentorch-gxpb2` remains open/in-progress because the full
size-gated AED/multishift geev dispatch is not implemented by this sub-lever.
