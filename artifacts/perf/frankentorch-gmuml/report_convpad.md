# frankentorch-gmuml partial mitigation: conv no-grad pads into a local Vec

## Target

`frankentorch-gmuml` (P1, profile-backed): the session autograd tape
(`TensorTape.nodes`) is append-only and never frees live nodes (index-based
`TensorNodeId` has no Drop), so repeated no-grad forwards on one session degrade
~linearly as retained intermediate tensors pile up and the heap grows. The
`conv1d/nograd_L` bench reuses ONE session across all iterations, so this
retention dominates the conv "gap" (gmuml/ejaga root-cause).

The full fix (RAII/refcounted tensor handles) is a core-engine rewrite. This is
mitigation (a) from the bead: the conv no-grad fast paths created a retained
`tensor_pad` SESSION node for the padded input (~4.2MB at L=1024, ~16MB at
L=4096 per call) and then copied it AGAIN via `tensor_values`. That retained
node leaks for the session's lifetime AND the second copy is pure waste.

## One Lever

In `functional_conv2d` (the conv1d→conv2d height-1 path too):

- Add `pad_nchw_zero_f64` / `pad_nchw_zero_f32`: zero-pad a contiguous NCHW
  buffer into ONE freshly-allocated buffer with a single copy of the source.
- No-grad f64 / f32 / generic fast paths now pad straight into a local `Vec`
  (no `tensor_pad` autograd node, one alloc + one copy instead of two).
- The padded autograd node is created LAZILY, only on the grad / composed paths
  (which need it so `tensor_pad`'s backward can un-pad). The no-grad paths return
  before reaching it, so they never retain a padded node.

No benchmark, API, dtype, shape, autograd-output, or error-surface change.

## Isomorphism Proof

- `pad_nchw_zero_*` writes the EXACT input values into the interior and `0.0`
  elsewhere — bitwise identical to `tensor_pad(_, [pad_w,pad_w,pad_h,pad_h], 0.0)`
  followed by reading the contiguous values. Same layout (NCHW), same zeros.
- Grad path is byte-for-byte unchanged (still uses the `tensor_pad` node).
- The `padding == 0` case reads `tensor_values(input)` directly, exactly as
  before (`padded == input`).
- Bit-exact verification: `cargo test -p ft-api conv` → 54 passed, 0 failed
  (conv1d/conv2d forward+grad+nograd match_torch goldens).
- Golden SHA-256: `sha256sum -c artifacts/optimization/golden_checksums.txt
  --ignore-missing` → all OK, including the two conv2d golden outputs.
- Full lib suite: 2048 passed, 1 pre-existing unrelated failure
  (`householder_product_ormqr_edge_cases_match_torch`, fails identically on HEAD
  — linalg, not conv).

## Same-worker A/B (vmi1156319, criterion --measurement-time 5 --sample-size 20)

`conv1d/nograd_L` (one session reused across all iters — exercises the
tape-growth degradation directly):

| shape | BEFORE (HEAD) median | AFTER median | speedup |
|-------|----------------------|--------------|---------|
| L/1024 | 31.7 ms [30.1, 33.3] | 24.5 ms [20.9, 28.8] | ~1.30x |
| L/4096 | 181.3 ms [158.9, 204.7] | 73.5 ms [68.8, 78.7] | ~2.47x |

AFTER L/4096 reproduced twice (71.5 ms, 73.5 ms) with ranges entirely below
BEFORE. The win scales with tensor size: bigger padded buffers = more retained
heap eliminated per call + bigger redundant copy removed. (One L/1024 AFTER run
showed 47.7 ms from peer contention — that run took 242 s wall vs 205 s; its low
matched HEAD's low, and run2 settled at 24.5 ms.)

## Score

L/4096: Impact 2.47 x Confidence 0.95 / Effort 0.4 = 5.9 (>= 2.0 keep gate).
Even the conservative L/1024 1.30x: 1.30 x 0.95 / 0.4 = 3.1.

## Verdict

Keep. Bit-exact, decisive same-worker win. Partial mitigation of gmuml for the
conv no-grad path; the general append-only-tape retention (all repeated no-grad
forwards) still needs the RAII/arena handle rewrite — gmuml stays open.
