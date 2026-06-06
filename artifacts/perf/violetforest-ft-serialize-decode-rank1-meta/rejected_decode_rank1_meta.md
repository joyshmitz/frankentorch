# ft-serialize decode rank-1 metadata fast path rejection

Date: 2026-06-06
Agent: VioletForest
Scope: `ft-serialize`

## Target

Post-half reprofile made native f64 decode the dominant serializer panel row:

```text
ts1 native_state_dict/decode_many_small_f64_1024x4: [334.98 us 339.78 us 344.32 us]
```

The target workload decodes 1024 native F64 tensors of shape `[4]`.

## Candidate

Temporarily added `TensorMeta::contiguous_1d` and routed native rank-1 tensor
metadata through it. The intent was to avoid the generic contiguous-stride and
shape-numel loops for the many-small-tensor decode path.

## Proof

Golden checksum:

```text
artifacts/optimization/golden_outputs/ft_serialize_decode_pass19.txt: OK
```

Focused proof test:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-serialize native_decode_many_small_f64_golden_summary_matches_fixture -- --nocapture
```

Result: passed.

Isomorphism:

Ordering: `BTreeMap::entry` insertion and duplicate-key detection remained
unchanged.

Tie-breaking: no comparator or tie-breaking path changed.

Floating point: payload decoding stayed on the existing `read_f64_payload` path.
No arithmetic or bit reinterpretation changed.

RNG: no RNG state or sampling path is involved.

Metadata: for rank-1 tensors, `TensorMeta::contiguous_1d(len, dtype, Cpu)`
constructed the same shape `[len]`, stride `[1]`, storage offset `0`, dtype,
device, quantization `None`, and `numel=len` that `TensorMeta::from_shape(vec![len], dtype, Cpu)`
would construct.

## Benchmark

Matched baseline with the committed generic rank decode source restored:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 5 --sample-size 20
vmi1149989 native_state_dict/decode_many_small_f64_1024x4: [246.33 us 253.45 us 261.01 us]
```

Candidate:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 5 --sample-size 20
vmi1149989 native_state_dict/decode_many_small_f64_1024x4: [242.44 us 257.34 us 270.37 us]
```

Same-worker delta: `253.45 us -> 257.34 us` p50, about 1.5 percent slower.

Score: Impact 0 * Confidence 5 / Effort 1 = 0.0, reject.

## Decision

Rejected. Source was reverted to the committed generic-rank decoder. No behavior
change remains.

Next deeper target: a batch/arena native decode primitive that reduces per-entry
allocation and tree-update overhead across many small tensors. Target ratio:
at least 2x on `decode_many_small_f64_1024x4`, likely by parsing metadata into a
scratch arena and constructing the returned map from a compact owned batch rather
than micro-tuning individual shape or payload loops.
