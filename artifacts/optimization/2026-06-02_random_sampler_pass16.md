# ft-data RandomSampler Pass 16

Bead: `frankentorch-snmk`

## Target

Fresh `ft-data` sampler target generated after ready perf beads were drained.
`RandomSampler::indices` without replacement repeatedly allocates a fresh
`0..size` vector for every full shuffled pass and for the remainder pass.

Criterion scenario:

```text
cargo bench -p ft-data --bench sampler_bench -- sampler/without_replacement_repeated_passes_4096x256 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

The workload samples `4096 * 256 + 1537` indices from a `size = 4096`
no-replacement sampler, forcing 256 full shuffled passes plus a remainder pass.

## Baseline

```text
worker: vmi1156319
sampler/without_replacement_repeated_passes_4096x256 time: [13.586 ms 13.997 ms 14.601 ms]
```

## Lever

Reuse one shuffle buffer and refill it to sequential `0..size` between passes,
then `extend_from_slice` into the result. This removes repeated heap allocation
while preserving the same Fisher-Yates shuffle calls.

## Isomorphism Proof

- RNG sequence unchanged: every full pass and the remainder pass call
  `shuffle` once on a `size`-length slice.
- Starting order before each shuffle unchanged: the attempted buffer was refilled
  to `0..size` before every pass after the first.
- Output ordering unchanged: full shuffled passes are appended in pass order, and
  the remainder still appends the first `remainder` shuffled indices.
- Replacement behavior untouched; the attempted lever only affected the
  no-replacement branch.
- Golden exact sequence fixture:

```text
827bc5a601bacca033c8127d6a1efe79711987bcaf41c789e6889166db100a52  artifacts/optimization/golden_outputs/random_sampler_pass16.txt
```

## Rebenchmark

```text
worker: vmi1293453
sampler/without_replacement_repeated_passes_4096x256 time: [4.1939 ms 4.3367 ms 4.4659 ms]
```

The middle estimate improved from `13.997 ms` to `4.3367 ms`, about 69.0 percent
faster, with a cross-worker caveat recorded in the confidence score.

## Validation

- `rch exec -- cargo test -p ft-data random_sampler -- --nocapture` passed on
  `vmi1156319`: 24 sampler-related tests passed.
- `rch exec -- cargo check -p ft-data --all-targets` passed on `vmi1149989`.
- `rch exec -- cargo clippy -p ft-data --all-targets --no-deps -- -D warnings`
  passed on `vmi1153651`; the dependency `ft-api` still emits its existing
  warning backlog.
- `rch exec -- cargo fmt -p ft-data --check` passed.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`
  passed.
- `git diff --check` passed.
- `ubs crates/ft-data/src/lib.rs crates/ft-data/benches/sampler_bench.rs
  crates/ft-data/Cargo.toml` still reports existing ft-data test unwrap,
  assertion, indexing, and false-positive secret-comparison inventory. The
  scanner's built-in format, clippy, check, and test-build subchecks passed.

## Decision

Kept.

Score: impact `3` x confidence `2` / effort `1` = `6.0`.

Kept diff: sampler scratch-buffer reuse, sampler benchmark case, exact-sequence
unit assertion, golden output fixture, and this evidence artifact.
