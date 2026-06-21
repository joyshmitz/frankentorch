# frankentorch-kgs4.144 BatchNorm2d f32 lazy-zero no-ship

- Bead: `frankentorch-kgs4.144`
- Agent: `IvoryDeer` / `cod-b`
- Target: `pytorch_gauntlet_bench` `gauntlet_batch_norm2d_f32_grad`, `[N,C,H,W]=[32,256,28,28]`, scalar-sum loss and backward.
- Lever attempted: add a lazy known-zero gradient representation to `ft-autograd` and make the f32 BatchNorm2d scalar-loss closures return lazy-zero `dx` and `dweight` plus dense `dbias`.
- Baseline RCH worker `vmi1149989`: ordinary `[60.183 ms, 63.388 ms, 65.717 ms]`; explicit scalar-sum `[54.504 ms, 58.024 ms, 62.578 ms]`; remote PyTorch failed because torch is unavailable on the worker.
- Local PyTorch sidecar: `/data/projects/frankentorch/.venv/bin/python`, Torch `2.12.0+cpu`, 32 threads, five 40-iteration totals median `0.311332728015 s`, or `7.783318 ms/iter`.
- Candidate RCH worker `vmi1153651`: ordinary `[101.81 ms, 105.48 ms, 109.54 ms]`; explicit scalar-sum `[65.028 ms, 68.775 ms, 74.333 ms]`; remote PyTorch failed because torch is unavailable on the worker.
- Candidate ratios vs local PyTorch: ordinary `13.55x` slower, explicit scalar-sum `8.84x` slower.
- Focused candidate tests: first run failed under the old materialized-residue contract; after temporary contract update, `cargo test -p ft-api functional_batch_norm2d_f32 --lib --profile release -- --nocapture` passed `6/0`.
- Final reverted-tree conformance: `rch exec -- cargo test -p ft-conformance --profile release` passed on `vmi1152480`.
- Verdict: rejected/reverted. Product source and temporary test-contract edits were manually restored; only this evidence bundle, Beads update, ledger, and scorecard should remain.
