# frankentorch-blt9 rejection report

## Target

- Bead: `frankentorch-blt9`
- Target: `[perf] ft-optim: remove AdamW persistent gradient clone`
- Worker: `ts1`
- Baseline command: `rch exec -- cargo bench -p ft-optim --bench optimizer_bench -- adamw/step_64x1024 --warm-up-time 1 --measurement-time 3 --sample-size 20`
- Candidate command: same command after adding a borrowed persistent-gradient parameter-update path.

## Result

Rejected. The candidate did not clear the Score >= 2.0 keep gate.

Criterion slope estimate, same worker `ts1`:

- Baseline: `342.932 us` with 95% CI `[335.509 us, 350.554 us]`
- Candidate: `335.543 us` with 95% CI `[324.554 us, 353.326 us]`
- Ratio from slope point estimates: `1.022x`

Criterion median/mean estimates were not supportive:

- Median: `336.010 us -> 339.185 us` (regression)
- Mean: `332.593 us -> 353.719 us` (regression)

Score: `Impact 1.02 x Confidence 0.50 / Effort 0.7 = 0.73`, below the keep threshold.

## Candidate Lever

The rejected candidate added a borrowed-gradient update API across `ft-autograd` and `ft-api`, then routed Adam and AdamW through that API to avoid cloning the persistent gradient buffer before updating parameters. The candidate source hunk was removed after the failed gate; the final source tree has no optimizer/API/autograd diff from the pre-candidate state.

## Isomorphism Proof

Final tree after rejection:

- `git diff --stat -- crates/ft-api/src/lib.rs crates/ft-autograd/src/lib.rs crates/ft-optim/src/lib.rs` is empty.
- `rch exec -- cargo test -p ft-optim adamw_ --lib` passed 14/14 on `ts1`.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed, including the existing AdamW golden outputs.

Candidate proof before rejection:

- Borrowed-gradient tape split test passed on `ts1`.
- AdamW filtered tests passed 14/14 on `ts1`.
- Ordering, tie-breaking, floating-point, and RNG behavior were unchanged by construction: the candidate preserved per-parameter iteration order, per-element operation order, Adam/AdamW bias-correction order, decoupled decay order, state-mismatch fail-closed checks, and did not introduce RNG.

## Evidence

- `baseline_estimates.json`
- `candidate_after_estimates.json`
- `final_adamw_tests.txt`
- `final_golden_check.txt`
- `final_source_diff.txt`

The original terminal tee files `baseline_adamw_step.txt`, `after_adamw_step.txt`, and `test_adamw_filtered.txt` are empty because RCH emitted its stream on stderr; the retained Criterion JSON estimate files are the authoritative benchmark evidence for this rejection.

## Next Primitive

Do not repeat this micro-lever. The next optimizer target should be a larger alien-graveyard primitive:

- Region/slab workspace for optimizer step transients and recurrent training loops, so parameter/gradient/state scratch is allocated once per optimizer instead of per step.
- Or a batched/vectorized optimizer execution path that treats optimizer state as contiguous SoA arrays and updates many parameters in cache-sized morsels.

Target ratio for the next primitive: `>=1.5x` on `adamw/step_64x1024` or a broader optimizer-training trace with allocation counters, with the same strict golden and rch same-worker gate.
