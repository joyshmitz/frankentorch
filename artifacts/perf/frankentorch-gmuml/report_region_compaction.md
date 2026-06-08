# frankentorch-gmuml: conv1d no-grad region compaction

## Target

- Bead: `frankentorch-gmuml`
- Source commit: `548b5a2b` (`perf(ft-api): conv1d no-grad forwards free their intermediates`)
- Benchmark: `cargo bench -p ft-api --bench ops_bench -- conv1d/nograd_L/4096 --warm-up-time 1 --measurement-time 5 --sample-size 20`
- Profile-backed failure: repeated no-grad `conv1d` forwards on one long-lived session retained reshapes and intermediate tensors, driving allocator/cache pressure as the tensor tape grew.

## Lever

Add a safe-Rust region compaction helper for no-grad outputs and apply it to batched `functional_conv1d`:

1. Capture `boundary = autograd_graph_node_count()` before the conv1d reshape/conv2d/reshape graph.
2. Run the existing conv1d route unchanged.
3. If the output does not require grad and is F64 or F32, materialize its values, truncate the tape back to `boundary`, and reinsert only the output leaf.
4. Leave grad outputs and unsupported dtypes untouched.

This preserves the returned tensor handle and its values while freeing per-call temporary graph nodes. It does not alter the convolution arithmetic.

## Benchmark Evidence

Pre-compaction Criterion baselines on `vmi1156319`:

| Run | Criterion interval |
| --- | --- |
| Baseline | `[90.414 ms 108.77 ms 130.58 ms]` |
| Clean baseline rerun | `[92.816 ms 109.28 ms 128.82 ms]` |

Clean HEAD confirmation after `548b5a2b`:

- `vmi1152480`: `[31.411 ms 32.219 ms 33.034 ms]`

The final clean HEAD run is cross-worker, so it is not used as a direct same-worker Criterion ratio. The keep gate for the committed lever is the worker-immune within-process degradation A/B recorded in `548b5a2b`:

- old path: late/early forward-time ratio `1.139` with `1202` retained nodes
- compacted path: late/early forward-time ratio `0.999` with `302` retained nodes

## Isomorphism Proof

- Ordering: output values are read from the already-computed contiguous output in the same order, then reinserted with the same shape.
- Floating point: no arithmetic is changed; the helper only copies F64 or F32 output storage after the original path computes it.
- RNG: no random state is read or advanced by the helper.
- Tie-breaking: not applicable to convolution.
- Autograd: the helper is skipped when the output requires grad, so backward graph behavior is unchanged for grad inputs. The no-grad output remains a leaf with `requires_grad=false`, matching the previous no-grad contract.
- Golden SHA-256: `sha256sum -c artifacts/optimization/golden_checksums.txt` passed.

Focused proof:

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-api functional_conv1d_nograd_compacts_intermediates_bit_exact -- --nocapture`
- Passed on clean HEAD on `vmi1152480`: bit-exact against the grad-capable route and retained exactly one output node above the boundary.

## Gates

- Passed on clean HEAD: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-api --all-targets` on `vmi1227854`
- Passed: `git diff --check`
- Passed: `sha256sum -c artifacts/optimization/golden_checksums.txt`
- Blocked by existing repo debt: `cargo clippy -p ft-api --all-targets -- -D warnings` reported 223 pre-existing `ft-api` lints outside this lever.
- Blocked by existing repo drift: `cargo fmt --package ft-api --check` produced broad formatting diffs across existing benches/examples and unrelated `src/lib.rs` regions.
- Unavailable: `ubs crates/ft-api/src/lib.rs` timed out after 180 seconds in the Rust scanner after producing only startup output.

## Score

- Impact: 3.0, because this removes retained temporary region nodes from the profile-backed `conv1d/nograd_L/4096` target and validates the deeper region-control primitive.
- Confidence: 0.85, because the worker-immune within-process proof removes the tape-growth slope and the focused proof is bit-exact; direct same-worker Criterion confirmation was not available through the current `rch` selector.
- Effort: 1.0, because the lever is small and scoped.
- Score: `3.0 * 0.85 / 1.0 = 2.55`, above the `>= 2.0` keep gate.

## Next Primitive

Generalize the same region/generation idea beyond `conv1d`: an explicit no-grad inference request boundary that can materialize live outputs and reclaim all temporary nodes without invalidating returned handles. This should attack the systemic append-only tape retention described by the bead, not another wrapper-local micro-path.
