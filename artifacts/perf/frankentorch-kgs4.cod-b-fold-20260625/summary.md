# frankentorch-kgs4 cod-b fold col2im lane-parallel win

Agent: PearlReef
Date: 2026-06-25

## Candidate

`tensor_fold` / `torch.nn.functional.fold` accumulates overlapping image patches
back into output images. The kept mainline implementation parallelizes independent
`(batch, channel)` output lanes. Each lane owns a disjoint contiguous
`output_h * output_w` block, so there is no cross-lane accumulation race. Inside
each lane, the block/kernel accumulation order stays identical to the serial
loop.

This matches the graveyard vectorized/morsel-driven parallelism route: split a
large irregular loop into independent cache-local morsels while preserving the
within-morsel arithmetic order.

## Head-To-Head Measurement

Current-main command after fast-forward to `61d6be15`:

```bash
AGENT_NAME=PearlReef \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
cargo run --release -p ft-api --example fold_h2h
```

Fixture: f64 no-grad `fold([64,576,2916] -> [64,64,56,56])`, five-iteration
minimum, local PyTorch CPU sidecar.

Measured:

| implementation | time | checksum |
| --- | ---: | ---: |
| FrankenTorch current main | 23.53 ms | `2.232285e5` |
| PyTorch | 142.09 ms | `2.232285e5` |

Ratio: FrankenTorch `6.04x FASTER`, checksum `MATCH`.

Earlier dirty-candidate command before the same source was fast-forwarded from
upstream:

```bash
AGENT_NAME=PearlReef \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
cargo run --release -p ft-api --example fold_h2h
```

Fixture: f64 no-grad `fold([64,576,2916] -> [64,64,56,56])`, five-iteration
minimum, local PyTorch CPU sidecar.

Measured:

| implementation | time | checksum |
| --- | ---: | ---: |
| FrankenTorch dirty candidate | 22.68 ms | `2.232285e5` |
| PyTorch | 137.16 ms | `2.232285e5` |

Ratio: FrankenTorch `6.05x FASTER`, checksum `MATCH`.

An uncaptured first run in the same session measured FT `22.66 ms` vs PyTorch
`166.31 ms`, FT `7.34x FASTER`, checksum `MATCH`; the captured rerun above is
the durable score.

## Additional Current-Main PyTorch Verification

After the fast-forward to `61d6be15`, PearlReef also reran the freshly landed
`tensor_unique` high-cardinality gate on the same warm `cod-b` target:

```bash
AGENT_NAME=PearlReef \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
cargo run --release -p ft-api --example unique_h2h
```

Results:

| fixture | FrankenTorch | PyTorch | ratio | correctness |
| --- | ---: | ---: | ---: | --- |
| few-unique-503 | 482.41 ms | 1557.96 ms | FT `3.23x FASTER` | `MATCH` |
| all-distinct | 197.14 ms | 1071.64 ms | FT `5.44x FASTER` | `MATCH` |

## Behavior Proof

Focused fold oracle command:

```bash
AGENT_NAME=PearlReef \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
cargo test -p ft-api fold_matches_torch_overlap_sum --lib -- --nocapture
```

Result: passed, `1 passed; 0 failed; 2375 filtered out`.

Current-main broader fold filter command:

```bash
AGENT_NAME=PearlReef \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
cargo test -p ft-api fold --lib -- --nocapture
```

Result: passed, `7 passed; 0 failed; 2369 filtered out`.

Current-main per-crate checks:

```bash
AGENT_NAME=PearlReef \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
cargo check -p ft-api --all-targets

AGENT_NAME=PearlReef \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
cargo clippy -p ft-api --all-targets -- -D warnings

AGENT_NAME=PearlReef \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
cargo test -p ft-conformance

AGENT_NAME=PearlReef \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
cargo test -p ft-api --lib
```

Results: check passed, clippy passed, conformance passed, and full `ft-api`
library tests passed (`2375 passed; 1 ignored`). A full
`cargo fmt -p ft-api --check` emitted broad pre-existing formatting churn across
many unrelated `ft-api` examples and unrelated `src/lib.rs` regions, so the
fold-specific formatting evidence is `rustfmt --edition 2024 --check
crates/ft-api/examples/fold_h2h.rs`, which passed.
