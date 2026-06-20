# Scorecard: frankentorch-kfdnn

Agent: IvoryDeer / cod-b
Phase: BOLD-VERIFY
Run: gauntlet_20260620T044748Z

## Summary

Result: no performance code kept.

The packed-u16 argmax sidecar was a radical layout/memory-traffic candidate. It passed focused correctness tests, but failed the target benchmark gate and was reverted before commit.

## PyTorch Head-to-Head

| Candidate state | FrankenTorch target median | PyTorch median | Ratio | Verdict |
| --- | ---: | ---: | ---: | --- |
| baseline | 5.6985 ms | 1.8203 ms | 3.130x slower | loss |
| packed-u16 trial | 5.7355 ms | 2.1262 ms | 2.697x slower | loss, no accepted gain |

Target win/loss/neutral ratio: 0 / 1 / 0.

## Correctness

Focused candidate tests passed before rejection:

- `ft-kernel-cpu::max_pool3d_sum_packed_u16_backward_matches_materialized_bits`
- `ft-api::functional_max_pool3d_sum_matches_pool_sum_backward_bits`
- `rch exec -- cargo test -p ft-conformance -- --nocapture`: passed on `vmi1149989` after the source revert.

The candidate source changes were reverted, so committed source remains at the pre-trial implementation.

## Verification Notes

- PyTorch oracle: `/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python`, torch `2.12.1+cpu`, 32 threads.
- `rch` was used for Rust correctness commands.
- The PyTorch benchmark had to run locally because the configured worker environment does not include the local PyTorch venv.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b` was used for all compile/bench commands.

## Next Highest-Value Lever

Do not retry sidecar-only variants for this bead. Follow-up `frankentorch-maxpool3d-scalar-loss-grad-buffers-7wru6` should attack autograd/session overhead: lazy gradient allocation, removing duplicate persistent gradient buffers, or a fused scalar-loss backward primitive with deterministic evidence intact.
