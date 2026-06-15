# frankentorch-ruphq rejection

Bead: `frankentorch-ruphq`

Candidate: collect `(String, DenseTensor)` pairs in the guarded rank-1 width-4 F64 native decode fast path, require strictly ascending payload keys, and build the final `BTreeMap` from the collected iterator.

## Profile-backed target

Baseline source state was unchanged after `frankentorch-f8uji` and `frankentorch-ykpvk`.

Completed RCH reprofile on `vmi1293453`:

`native_state_dict/decode_many_small_f64_1024x4`

`[254.89 us 260.78 us 270.87 us]`

Fresh same-worker baseline attempts on `vmi1293453` went progress-stale before a Criterion row and were canceled as non-evidence.

## Behavior proof while candidate was present

- RCH `ovh-a`: `cargo test -j 1 -p ft-serialize native_ -- --nocapture` passed `19/19`, including the new unsorted-key fast-path decline test.
- Local `cargo fmt -p ft-serialize --check` passed after source removal.

## Benchmark result

The candidate rebench selected `vmi1293453` but went progress-stale before a Criterion row. It was canceled as unusable evidence.

Verdict: REJECT / NO KEEP. Score `0.0` because no completed rebench row exists.

## Source state

The candidate source hunk and focused test were removed. Final source has no `ruphq` diff.

## Reroute

Do not retry a simple sorted-pair `BTreeMap` bulk-build family without a more reliable same-worker bench harness. The next ft-serialize route should move deeper into fixed-layout native parsing or a dedicated state-dict construction path that reduces key/string/map materialization while still returning a normal `BTreeMap` with identical ordering and duplicate-key behavior.
