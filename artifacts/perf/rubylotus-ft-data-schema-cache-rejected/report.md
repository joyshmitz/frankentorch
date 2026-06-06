# frankentorch-w1vn ft-data TensorDataset schema-cache rejection

Agent: RubyLotus
Crate: `ft-data`
Outcome: rejected, no source change kept

## Profile-backed target

The target was `DataLoader` collation for immutable `TensorDataset` inputs. The intended deeper
primitive was cache/locality layout: infer a homogeneous tensor schema at `TensorDataset`
construction time, then skip repeated per-batch name, shape, and value-length validation during
collation.

Fresh baseline:

```text
worker: vmi1149989
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-data --bench dataloader_bench -- --warm-up-time 1 --measurement-time 3 --sample-size 10

dataloader/epoch_2048x256_batch128        [256.75 us 296.32 us 320.41 us]
sampler/without_replacement_size4096...   [131.13 us 135.39 us 138.83 us]
```

Prior same-worker pre-edit baseline from the current pass sequence:

```text
worker: ts1
dataloader/epoch_2048x256_batch128        [494.96 us 605.68 us 663.75 us]
sampler/without_replacement_size4096...   [135.79 us 137.13 us 138.54 us]
```

## Attempted Lever

The draft added a private `TensorItemSchema`, cached an optional homogeneous schema in
`TensorDataset`, routed `collate_indices` through a schema fast path, and fell back to the existing
validating collation path when construction detected heterogeneous or invalid items.

## Rebenchmark

Same-worker decision run:

```text
worker: ts1
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-data --bench dataloader_bench -- --warm-up-time 1 --measurement-time 3 --sample-size 10

dataloader/epoch_2048x256_batch128        [517.68 us 619.17 us 667.82 us]
sampler/without_replacement_size4096...   [131.28 us 132.66 us 133.58 us]
```

## Decision

Rejected. The same-worker dataloader median regressed:

- `dataloader/epoch_2048x256_batch128`: 605.68 us -> 619.17 us
- Score: below keep threshold because the measured impact is negative

The source hunk was manually removed. `cargo fmt -p ft-data --check` passes with no `ft-data`
source diff remaining.

## Isomorphism Note

The draft preserved observable behavior by activating the fast path only after construction had
proved all samples shared tensor counts, names, shapes, and value lengths. Invalid or heterogeneous
datasets fell back to the existing validator and error strings. Batch order, tensor order, copied
values, sampler RNG, and shape construction were unchanged.

The rejection means the next ft-data pass should move deeper than validation hoisting, likely toward
a genuinely different layout primitive such as columnar/contiguous `TensorDataset` storage that
avoids per-sample `DataItem` object traversal during hot collation.
