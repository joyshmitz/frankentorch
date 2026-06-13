# frankentorch-gxpb2 pass 2 q_acc supplement

Date: 2026-06-13
Bead: `frankentorch-gxpb2`
Source commit: `9beeefac`

## Why this supplement exists

The committed keep report used the original `hz1` before/after pair. During
post-commit verification, the attempted pinned `hz1` rerun selected
`vmi1149989`, so a stricter same-worker clean-head comparison was captured for
the same source lever.

## Same-worker A/B

Clean-head behavior was restored temporarily in the working tree and benchmarked
on the same worker as the candidate run. The candidate source was then restored.

| row | worker | before | after |
| --- | --- | ---: | ---: |
| `eigvals_f64_256x256` | `vmi1149989` | `[38.311 ms 40.828 ms 45.699 ms]` | `[22.106 ms 25.710 ms 30.316 ms]` |

Median speedup: `40.828 / 25.710 = 1.588x`.

This is still the same single lever: skip materializing the dead `q_acc`
identity matrix when `want_vectors == false`. Full `eig` still receives the
identity Q accumulator.

## Isomorphism

- Ordering preserved: yes; no Hessenberg or Francis loop order changed.
- Tie-breaking unchanged: yes; no threshold, selected-`m`, deflation, or complex
  pair slot policy changed.
- Floating-point unchanged: yes for all live eigenvalue arithmetic; the skipped
  vector is a guarded sink in values-only calls.
- RNG unchanged: no RNG.
- Golden output: strict stdout SHA stayed
  `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.

## Extra gates

- `cargo test -j 1 -p ft-kernel-cpu --lib eig -- --nocapture`: passed `24/24`
  on `vmi1149989`.
- `cargo check -j 1 -p ft-kernel-cpu --all-targets`: passed on `vmi1149989`.
- `cargo clippy -j 1 -p ft-kernel-cpu --all-targets -- -D warnings`: passed on
  `vmi1149989`.
- `cargo fmt -p ft-kernel-cpu --check`: RCH refused non-compilation remote
  fallback; local check passed.
- `ubs crates/ft-kernel-cpu/src/lib.rs`: exit `0`; no critical issues.

## Route

Keep `9beeefac`. `frankentorch-gxpb2` remains in progress because this proof
does not implement the actual size-gated AED/multishift geev dispatch.
