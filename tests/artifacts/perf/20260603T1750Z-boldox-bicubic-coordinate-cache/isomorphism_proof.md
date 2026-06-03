# frankentorch-5ys7 Isomorphism Proof

## Change

`interpolate_bicubic` now precomputes x/y cubic tap indices and weights once per output coordinate, then reuses those plans across every batch-channel plane.

## Invariants

- Ordering preserved: output rows remain row-major by `plane`, `oy`, then `ox`.
- Tap order preserved: accumulation remains `dy=0..4` outside `dx=0..4`, matching original `dy=-1..=2` then `dx=-1..=2`.
- Floating-point preserved: `src`, `floor`, `frac`, `cubic_weight`, and clamp inputs are unchanged; each output still evaluates `wy * wx * storage[...]` in the same left-associative order.
- Tie-breaking unchanged: no ordering/tie path exists.
- RNG unchanged: no RNG path exists.
- DType/shape unchanged: returns the same F64 tensor shape `[batch, channels, oh, ow]`.

## Evidence

- Baseline: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-api --bench ops_bench -- interpolate_bicubic/8x32x64x64_2x --warm-up-time 1 --measurement-time 5 --sample-size 10` on `vmi1156319`: `[52.242 ms 56.937 ms 60.054 ms]`.
- Proof test: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-api interpolate_bilinear_bicubic_parallel_match_serial_bit_exact -- --nocapture` passed on `vmi1293453`.
- After: same Criterion target on `vmi1156319`: `[17.611 ms 18.827 ms 20.476 ms]`.
- Golden outputs: `sha256sum -c tests/artifacts/perf/20260603T1750Z-boldox-bicubic-coordinate-cache/golden_checksums.txt` passed.
- Score: impact 5 x confidence 4 / effort 2 = 10.0.
