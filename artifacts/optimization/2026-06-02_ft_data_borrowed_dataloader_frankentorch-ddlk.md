# frankentorch-ddlk ft-data DataLoader Borrowed-Sample Pass

## Target

- Bead: `frankentorch-ddlk`
- Scenario: `dataloader/epoch_2048x256_batch128`
- Baseline command: `rch exec -- cargo bench -p ft-data --bench dataloader_bench -- dataloader/epoch_2048x256_batch128 --warm-up-time 1 --measurement-time 5 --sample-size 20`
- Baseline worker/result: `vmi1227854`, `[2.7363 ms 3.0361 ms 3.3085 ms]`

## Lever Attempted

Avoid cloning every `TensorDataset` sample before collation by adding an opt-in borrowed-sample path and collating `&DataItem` references. A first draft measured well but changed the fallback timing for custom datasets. The draft was tightened so only explicitly borrow-capable datasets used the fast path and non-borrowing datasets kept the old `validate -> get -> push` loop ordering.

UBS also flagged three non-secret equality comparisons as critical. Those were temporarily rewritten to scanner-clean forms, then reverted with the source lever when the pass failed benchmark proof.

## Isomorphism Proof Considered

- Ordering preserved: the tightened borrow path iterated the same `indices[start..end]` order, and the fallback kept the old per-index validation/fetch order.
- Tie-breaking unchanged: no comparator, sorting, or sampler tie-breaking changed.
- Floating-point unchanged: values were copied in the same order; no arithmetic was added to collation.
- RNG unchanged: shuffle state, sampler order, and seeds were untouched.
- Golden output: `ed95568ba46c9d70b871772ca0937890ad2f9f418dc775925b341829540f6dce`.

## Verification While Lever Was Applied

- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`: passed.
- `rch exec -- cargo test -p ft-data dataloader -- --nocapture`: passed, 23 tests.
- `rch exec -- cargo check -p ft-data --all-targets`: passed.
- `rch exec -- cargo clippy -p ft-data --all-targets --no-deps -- -D warnings`: passed.
- `ubs --only=rust crates/ft-data/src/lib.rs`: passed after scanner-clean equality rewrites.

## Benchmark Results

- Draft after-run on `vmi1227854`: `[1.3553 ms 1.4576 ms 1.5905 ms]`.
- Tightened after-run on `vmi1293453`: `[760.02 us 794.56 us 828.43 us]`.
- Final post-UBS after-run on `vmi1153651`: `[3.5154 ms 3.6208 ms 3.7579 ms]`, with two outliers and concurrent fleet activity on the selected worker.

## Decision

Rejected. The final proof run did not confirm a real win against the baseline, and the RCH fleet was heavily contended during the decisive measurement. Impact `0` x confidence `1` / effort `2` = `0.0`, below the required `2.0` threshold.

The ft-data source lever was reverted. The golden fixture and this artifact remain as negative-result evidence for a future same-worker or lower-contention rerun.
