# frankentorch-kgs4.118 - conv3d all-ones dout backward fast path

Date: 2026-06-18
Agent: cod-a / IvoryDeer
Status: measured keep; PyTorch loss remains

## Target

Current routing evidence from `artifacts/perf/frankentorch-next-reprofile-20260617/current_top_train_reprofile.log`:

- `conv3d/grad`: median about 41.312 ms.
- The benchmark loss is `tensor_sum(out)`, so the upstream gradient entering conv3d backward is exactly `+1.0` at every output element.

This was routing evidence only until the 2026-06-20 gauntlet closeout below.

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
- Benchmark status: KEEP after 2026-06-20 same-worker `ovh-a` Criterion proof:
  parent baseline `29.723 ms`, current `26.595 ms`, `1.12x` faster / `10.5%`
  lower median, with non-overlapping intervals. Local PyTorch CPU comparator for
  the same f64 shape measured `7.593859 ms`, so FrankenTorch still loses `3.50x`.
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

## 2026-06-20 Gauntlet Closeout

- Baseline worktree: detached parent `870abe0d` (`75d87600^`) at
  `/data/projects/frankentorch-kgs4-118-baseline`.
- Baseline command: `RCH_WORKER=ovh-a RCH_WORKERS=ovh-a
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec --
  cargo bench -p ft-api --bench ops_bench -- conv3d/grad --noplot`.
- Baseline result: selected `ovh-a`; `conv3d/grad [29.423 ms 29.723 ms 30.038 ms]`.
- Current command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a
  rch exec -- cargo bench -p ft-api --bench ops_bench -- conv3d/grad --noplot`.
- Current result: selected `ovh-a`; `conv3d/grad [26.116 ms 26.595 ms 27.077 ms]`.
- PyTorch comparator: local CPU PyTorch `2.12.1+cpu`, 32 compute threads and 32
  interop threads, same f64 `[2,32,8,16,16] @ [32,32,3,3,3]` stride1/pad1
  sum-loss backward; median `7.593859 ms`.
- Verdict: keep the existing source change and close `frankentorch-kgs4.118` as a
  measured internal win, but classify the row as a PyTorch loss (`0W / 1L / 0N`).
- Validation: `ft-kernel-cpu conv3d` 2/0, `ft-api conv3d` 10/0, strict scheduler
  conformance 1/0.
- Evidence scorecard:
  `artifacts/perf/frankentorch-kgs4.118/gauntlet_20260620T0108Z/SCORECARD.md`.
