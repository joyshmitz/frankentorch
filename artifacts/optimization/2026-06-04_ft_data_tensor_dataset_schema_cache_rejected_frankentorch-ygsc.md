# ft-data TensorDataset Schema Cache Rejection (frankentorch-ygsc)

## Target

- Skill loop: `/repeatedly-apply-skill` pass 63 with `/extreme-software-optimization`, `/alien-graveyard`, and `/alien-artifact-coding`.
- Fallback bead: `frankentorch-ygsc`, created because `br ready --json` had no ready perf beads and active perf surfaces were already assigned to other agents.
- Benchmark surface: `cargo bench -p ft-data --bench dataloader_bench -- dataloader/epoch_2048x256_batch128 --warm-up-time 1 --measurement-time 5 --sample-size 20`.

## Baseline

RCH Criterion baseline on worker `ts1`:

- `dataloader/epoch_2048x256_batch128`: `[1.0850 ms 1.2906 ms 1.5875 ms]`.
- Adjacent sampler context: `sampler/without_replacement_size4096_samples66560`: `[126.96 us 129.79 us 131.87 us]`.

The profile-backed target was the `TensorDataset` batching path: each batch revalidates sample tensor count, names, shapes, and value lengths even when the dataset is immutable and uniform.

## Alien Primitive Considered

The harvested primitive was a staged validation/schema-cache variant of the graveyard's cache/locality and validation-membrane ideas: prove an immutable dataset schema once at construction, then let batch collation use the cached shape/name/value-length metadata instead of repeating the same validation every batch.

## Lever Attempted

One source lever was attempted and then removed:

- Add a private cached `TensorDatasetTensorSchema`.
- Infer it in `TensorDataset::new` only when all samples have identical tensor counts, names, shapes, and value lengths.
- Use the cached schema in `collate_indices` only for proven-uniform datasets.
- Keep nonuniform or malformed datasets on the existing validation path.

## Behavior Proof

Candidate-only proof passed before rejection:

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-data tensor_dataset_schema_cache -- --nocapture`
- Worker: `vmi1264463`.
- Result: `2 passed; 0 failed`.
- Candidate golden output sha256: `8aab68fd6c3bd0aac8fe0b7242a246d90ab0ad3fe11c26adeeb77f53ed5e2a1f`.

Isomorphism obligations for the candidate:

- Batch index order unchanged: the fast path still iterated `indices` in order.
- Tensor creation order unchanged: outer loop remained tensor-index order from the first sample.
- Error behavior unchanged for nonuniform datasets: schema inference returned `None`, preserving the old per-batch validation path.
- Floating-point/RNG unchanged: no arithmetic or sampling changed; values were copied byte-for-byte through `extend_from_slice`.
- Tie-breaking unchanged: no comparator, threshold, or ordering relation changed.

## Rebenchmark

RCH Criterion after run on worker `ts2`:

- `dataloader/epoch_2048x256_batch128`: `[2.4152 ms 2.5082 ms 2.5797 ms]`.

This did not beat the existing rch baseline and therefore failed the performance gate. A same-worker diagnostic baseline attempt was started, but the temporary source state was invalid and produced no usable baseline, so no favorable performance claim was retained.

## Score

- Impact: 0, because no real win was proven.
- Confidence: 4, because correctness proof passed but performance evidence was negative/inconclusive.
- Effort: 1.
- Score: `0 x 4 / 1 = 0.0`.

## Disposition

- Source/test/golden-checksum hunk removed.
- Candidate golden file retained only as negative-result evidence.
- No runtime behavior change kept.
- No-ceiling pivot: stop dataloader validation micro-tuning. Next profile pass should target a structurally different hot primitive, preferably a high-cost tensor/kernel path or a data pipeline allocation/layout primitive with a same-worker rch baseline.
