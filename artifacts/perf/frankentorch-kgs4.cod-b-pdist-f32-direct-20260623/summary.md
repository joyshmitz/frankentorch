# frankentorch-kgs4 cod-b pdist f32 direct-kernel no-ship

Agent: QuietMeadow
Date: 2026-06-23
Target: no-grad `tensor_pdist(x, p=2)` f32, shape `512x64`
Crate scope: `ft-api`
Warm target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b`

Tracker note: `br ready --json` and `br list --status in_progress --json` failed
with duplicate id `frankentorch-kgs4.150`; selection proceeded from
`bv --robot-triage`.

Current shipped route:

- RCH worker `ovh-a`
- Command: `cargo bench -p ft-api --bench cdist_bench -- pdist_f32_p2_mm/512x64 --warm-up-time 2 --measurement-time 6 --sample-size 20 --noplot`
- Pre-candidate interval: `[881.65 us 887.85 us 894.61 us]`
- Post-revert interval: `[891.74 us 896.33 us 900.84 us]`
- Criterion: no change in performance detected.

PyTorch comparator:

- torch `2.12.0+cpu`, 32 threads
- `min=0.042981 ms`, `median=0.044830 ms`, `p95=0.050466 ms`
- checksum `883173.937500`
- Post-revert FT/PyTorch ratio: `20.86x SLOWER`

Candidate tried and reverted:

- Direct strict-upper-triangle f32 p=2 row-pair kernel using `f32x8`
- RCH worker `vmi1149989`
- Initial interval: `[1.0313 ms 1.0905 ms 1.1573 ms]`
- Repeat interval: `[785.43 us 881.74 us 1.0271 ms]`
- Criterion repeat: no change in performance detected (`p = 0.26`)
- Candidate ratio by repeat midpoint: `20.51x SLOWER` vs PyTorch min, not a
  reliable/statistically significant keep.

Final state: source reverted; only evidence committed.

Green gate:

- `cargo test -p ft-api pdist_p2_f32_fused_nograd_matches_composed_path --lib -- --nocapture`
- RCH worker `ovh-a`
- `1 passed; 0 failed; 2372 filtered out`
