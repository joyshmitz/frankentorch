# ft-serialize Sorted Native Decode Fast Path Rejection (frankentorch-a6mg)

## Target

- Skill loop: `/repeatedly-apply-skill` pass 64 with `/extreme-software-optimization`, `/alien-graveyard`, and `/alien-artifact-coding`.
- Fallback bead: `frankentorch-a6mg`, created because `br ready --json` had no ready perf beads and active perf beads were owned by other agents.
- Benchmark surface: `cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 5 --sample-size 20`.

## Baseline

RCH Criterion baseline on worker `ts2`:

- `native_state_dict/decode_many_small_f64_1024x4`: `[483.42 us 484.65 us 485.96 us]`.

Full same-run residual context:

- `save_single_f32_1m`: `[1.3638 ms 1.3671 ms 1.3702 ms]`.
- `save_single_f64_1m`: `[1.4310 ms 1.4340 ms 1.4366 ms]`.
- `save_single_f16_1m`: `[1.6277 ms 1.6299 ms 1.6320 ms]`.
- `save_single_bf16_1m`: `[1.6675 ms 1.6694 ms 1.6717 ms]`.

Prior evidence ruled out repeating the obvious f64 chunk writer and native header-read micro-levers, so this pass targeted a different structural primitive.

## Lever Attempted

Candidate-only source hunk, now removed:

- Keep strictly increasing native state-dict keys in a `Vec<(String, DenseTensor)>`.
- Build the output `BTreeMap` after payload parsing for sorted unique input.
- Preserve immediate duplicate-key errors for equal adjacent keys.
- Fall back to the existing `BTreeMap::entry` parser path at the first out-of-order key, before parsing that tensor's shape or payload.

## Behavior Proof

Candidate-only proof:

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-serialize native_decode -- --nocapture`
- Worker: `ts2`.
- Result: `native_decode_many_small_f64_golden_summary_matches_fixture` passed.
- Existing decode fixture sha256: `93332e6e21332f43535b64ab5fe3224f22213becb95cb4d3b6e0ed9888dbe943`.

Isomorphism obligations for the candidate:

- Tensor payload parse order unchanged for sorted inputs.
- Returned map key order unchanged because the public type remains `BTreeMap`.
- Duplicate key error text unchanged; duplicate detection remained before shape/payload parsing.
- Out-of-order inputs fell back to the original entry-based duplicate detection path before current-tensor parsing.
- Floating-point bit patterns unchanged: decode still used the same `read_f64_payload` conversion.
- RNG/tie behavior unchanged: no random state, comparator tie-breaker, or arithmetic reassociation changed.

## Rebenchmark

RCH Criterion after run on the same worker `ts2`:

- `native_state_dict/decode_many_small_f64_1024x4`: `[703.21 us 704.42 us 705.64 us]`.

The sorted vector path was substantially slower than the baseline, likely because delaying `BTreeMap` construction did not remove the final tree-build cost and added branch/state overhead in the parser.

## Score

- Impact: 0, because the measured result regressed.
- Confidence: 5, because baseline and after were on the same worker and the focused proof passed.
- Effort: 2.
- Score: `0 x 5 / 2 = 0.0`.

## Disposition

- Runtime source hunk removed.
- No golden checksum change kept.
- Negative-result artifact kept.
- No-ceiling pivot: stop native-decode map-construction tuning. The next serialization attack should use a fundamentally different primitive, such as arena/slab allocation for many small tensors or a format-level batch metadata layout, only after a fresh rch profile-backed baseline.
