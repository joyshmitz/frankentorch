# frankentorch-z5sey: owned f64 slab native decode candidate rejected

## Target

- Bead: `frankentorch-z5sey`
- Hot row: `native_state_dict/decode_many_small_f64_1024x4`
- Candidate primitive: decode all native F64 tensor payloads into one owned slab and materialize tensors with offset metadata and shared immutable storage.

## Baseline

- RCH worker: `vmi1149989`
- Command: `cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 5 --sample-size 20`
- Artifact: `artifacts/perf/frankentorch-z5sey/baseline_decode_many_small_f64_1024x4.log`
- Result: `[361.48 us 400.87 us 445.68 us]`

## Candidate Evidence

- First candidate test compile failed on ambiguous float literal in the new slab test.
  - Artifact: `artifacts/perf/frankentorch-z5sey/test_native_slab_candidate.log`
- Fixed candidate native tests passed on `vmi1149989`.
  - Artifact: `artifacts/perf/frankentorch-z5sey/test_native_slab_candidate_retry.log`
  - Result: `16 passed; 0 failed`
- Golden SHA check passed for all present fixtures, including `ft_serialize_decode_pass19.txt`.
  - Artifact: `artifacts/perf/frankentorch-z5sey/golden_sha256_slab_candidate.log`
- Candidate benchmark attempt on `vmi1149989` did not produce a Criterion timing line and was canceled after it became stale.
  - RCH build: `29884606035530929`
  - Artifact: `artifacts/perf/frankentorch-z5sey/candidate_slab_decode_many_small_f64_1024x4.log`

## Verdict

- Reject.
- Score: `0.0`.
- Reason: no valid same-worker candidate timing was produced, so the lever cannot satisfy the Score >= 2.0 keep rule. The source hunk was removed.
- Additional design risk: a fully sound shared-slab representation needs tensor-local public storage semantics across `storage()`, `view()`, and `typed_storage()`/typed storage views. The rejected draft proved contiguous read and copy-on-write behavior, but the public storage-view surface is wider than this one decode hunk should alter.

## Next Route

Attack a deeper storage representation primitive rather than retrying decode micro-levers:

- introduce an explicit tensor-local typed storage view API, or
- introduce small/inline fixed-width F64 tensor storage for width-4 native tensors with unchanged public accessors, or
- redesign `TensorStorage` around slice-aware backing storage across all call sites.

Do not retry key materialization, sorted-map construction, delayed materialization, rank-1 metadata, scalar payload collection, or the rejected shared `Arc<Vec<f64>>` slab shape.
