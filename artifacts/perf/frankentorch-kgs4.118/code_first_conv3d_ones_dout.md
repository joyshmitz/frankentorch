# frankentorch-kgs4.118 - conv3d all-ones dout backward fast path

Date: 2026-06-18
Agent: cod-a / IvoryDeer
Status: code-first batch-test pending

## Target

Current routing evidence from `artifacts/perf/frankentorch-next-reprofile-20260617/current_top_train_reprofile.log`:

- `conv3d/grad`: median about 41.312 ms.
- The benchmark loss is `tensor_sum(out)`, so the upstream gradient entering conv3d backward is exactly `+1.0` at every output element.

This is routing evidence only. The campaign still needs same-worker criterion proof against the legacy original after the batch test window.

## Lever

Specialize `conv3d_backward_f64` for non-empty `dout` slices where every value is exactly `+1.0`.

The generic path materializes:

- `dout_flat[flat, out_ch]`, all ones for a sum-loss backward.
- `dout_t[out_ch, flat]`, also all ones.
- `dpanel[flat, patch_width]`, where every row is the same sum over `weight_flat`.

The specialized path collapses those repeated matrices:

- Compute one `dweight_row = ones[1, flat] @ panel[flat, patch_width]`.
- Copy that row to every output channel's dweight row.
- Compute one `dpanel_row = ones[1, out_ch] @ weight_flat[out_ch, patch_width]`.
- Scatter-add that repeated row through a col2im specialization instead of materializing `flat * patch_width` identical rows.
- Return `dbias = flat` for every output channel when bias is present.

Non-ones `dout` and empty `dout` stay on the generic implementation.

## Guard

Added `conv3d_backward_ones_dout_fast_path_matches_generic_reference` in `crates/ft-kernel-cpu/src/lib.rs`.

The guard compares the public fast path against the unchanged private generic reference on a multi-batch, multi-channel, strided 3D convolution shape with bias. The comparison uses a tight floating tolerance to avoid coupling correctness to the matrixmultiply crate's row tiling choices.

## Negative-Evidence Ledger

Attempt: `frankentorch-kgs4.118`

- Lever: conv3d sum-loss backward all-ones `dout` collapse.
- Expected win class: bandwidth and allocation reduction in realistic training traces where scalar sum loss drives conv3d backward.
- Correctness risk: a missed non-ones upstream gradient would be catastrophic; mitigated by exact `+1.0` bit trigger and generic fallback.
- Benchmark status: pending. Do not mark as a keep until same-worker criterion shows a real `conv3d/grad` win and conformance stays green.
- Revert predicate: any conformance failure, measurable `conv3d/grad` regression, or benchmark evidence showing the `dout` scan costs more than the repeated-matrix collapse saves.

Avoided dead ends from the campaign ledger:

- Did not retry max_pool2d borrowed-input or direct 2x2 side paths; prior artifacts already rejected or narrowed those lanes.
- Did not retry RMS/layer/group norm unit-dy paths; those belong to normalization beads with separate gradient semantics.
- Did not touch conv3d forward streaming; that previous lever was already distinct from this backward-only sum-loss specialization.

## Alien / Optimization Scorecard

- Cache-aware layout: eliminates two large repeated intermediate matrices from the sum-loss case.
- Branchless/SIMD/codegen: not used in this batch; the faster lever is structural redundancy removal.
- Arena/custom allocator: not used; no new long-lived allocation policy.
- EV score: impact 4, confidence 3, effort 2 -> 6.0.

## Required Follow-Up

- Run the campaign batch criterion comparison for `conv3d/grad` against the original legacy oracle.
- Run conformance for conv3d backward and full affected training-gradient lanes.
- Update this ledger with measured keep/reject evidence before closing the bead.
