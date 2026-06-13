# frankentorch-bw61h pass 3 same-order tiled sweep rejection

Bead: `frankentorch-bw61h`
Lever: guarded production helper for the Francis QR bulge row/column update.
Verdict: REJECTED; source hunk removed.

## Candidate

The candidate hoisted the production row/column update in `eig_francis_schur_traced`
into a same-order helper and walked fixed-size row and column tiles. It preserved
the original row sweep before column sweep, preserved ascending `j`/`i` order
inside each tile, did not change shift selection, and did not change public API
or RNG behavior.

## Behavior proof

- Focused shadow profile tests:
  `cargo test -p ft-kernel-cpu --lib eig_francis_shadow_profile -- --nocapture`
  passed `3/3` on `vmi1167313`.
- Duplicate focused test with `-j 1` passed `3/3` on `vmi1149989`.
- Broader eig filter:
  `cargo test -j 1 -p ft-kernel-cpu --lib eig -- --nocapture`
  passed `24/24` on `vmi1227854`.
- Strict `eigvals_golden` extracted stdout SHA-256:
  `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.
- The printed n=64/n=128/n=256 `eigvals` and `eig` digests stayed pairwise
  identical to the pass-1 baseline.
- Local `cargo fmt -p ft-kernel-cpu --check` passed after the source hunk was
  removed.
- Post-removal sanity check:
  `cargo test -j 1 -p ft-kernel-cpu --lib eig_francis_shadow_profile -- --nocapture`
  passed `3/3` on `vmi1227854` with source restored.
- UBS on `crates/ft-kernel-cpu/src/lib.rs` reported no critical issues; it
  still reports the existing broad warning inventory for this large file.

## Same-worker benchmark gate

Comparable worker: `vmi1227854`.

- Pass-1 baseline `eigvals_f64_256x256`:
  `[25.356 ms 26.137 ms 26.577 ms]`
- Candidate `eigvals_f64_256x256`:
  `[27.396 ms 28.766 ms 29.930 ms]`

Median result: `28.766 / 26.137 = 1.101x` slower. This fails the Score gate.
The candidate was rejected without running the `eig_f64_256x256` after row
because the shared values-only path had already regressed on the same worker.

## Score

`0.0 = negative Impact * Confidence / Effort`.

## Reroute

Do not repeat same-order row/column tiling for this Francis path. The next
candidate must be a deeper primitive: a private grouped operation-tape proof
for BLAS-3-style far updates, a strict-fallback Schur-window kernel, or an
AED-derived shift-list route with explicit shift/deflation/eigen-slot proof.
