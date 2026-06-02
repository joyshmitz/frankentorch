# ft-optim AdamW First-Step Fusion Attempt

- Bead: `frankentorch-wxtp`
- Parent umbrella: `frankentorch-kgs4`
- Skills: `/profiling-software-performance`, `/extreme-software-optimization`, `/alien-graveyard`, `/alien-artifact-coding`
- Crate: `ft-optim`
- Target benchmark: `adamw/step_64x1024`
- Outcome: rejected, no source change kept

## Profile Target

AdamW first-step on 64 parameters of 1024 f64 values remained a visible
optimizer-step hotspot. The current code builds `next_m` and `next_v` in two
passes and then walks both to update parameter values.

Baseline:

```text
worker: vmi1293453
command: rch exec -- cargo bench -p ft-optim --bench optimizer_bench -- adamw/step_64x1024 --warm-up-time 1 --measurement-time 5 --sample-size 20
adamw/step_64x1024: [420.57 us 428.82 us 435.49 us]
```

Attempted after:

```text
worker: vmi1293453
command: rch exec -- cargo bench -p ft-optim --bench optimizer_bench -- adamw/step_64x1024 --warm-up-time 1 --measurement-time 5 --sample-size 20
adamw/step_64x1024: [562.28 us 568.71 us 576.39 us]
```

Delta:

- p50: `428.82 us -> 568.71 us`, about 32.6% slower.
- Score: Impact -1 x Confidence 4 / Effort 1 = -4.0.
- Keep decision: rejected and source lever reverted.

## Alien Recommendation Card

Attempted primitive: loop fusion plus allocation-pressure reduction for the
AdamW first-step path where both moment buffers are absent.

Mapped graveyard sections:

- Vectorized/morsel execution: keep independent per-element work in a single
  tight loop when it reduces memory traffic.
- Artifact contract: every optimization must carry a benchmark comparator,
  proof obligations, and a fallback.
- Constants/cache warning: an apparently better layout can lose if it hurts
  optimizer-vectorization or adds branch pressure.

Expected value after measurement: below threshold. The fallback was applied:
keep the existing multi-pass AdamW implementation.

## Isomorphism Proof For The Rejected Draft

The draft was behavior-equivalent but slower:

- Ordering: parameter indices, moment-buffer indices, and state commit order
  remained unchanged.
- Floating point: each element used the same formulas for `m`, `v`, bias
  correction, `sqrt`, decoupled weight decay, and parameter subtraction. The
  proof test compared all parameter/moment bits against an inline copy of the
  original multi-pass math.
- Errors: hyperparameter checks, gradient length checks, state length checks,
  and `tensor_update_param_values` remained before optimizer-state commit.
- Tie-breaking/RNG: AdamW has no tie-breaking or RNG.
- Golden output: the rejected draft's proof fixture is pinned by sha256
  `0e1ce8b214f41eb3cd8ef84793006256e03688f0f56763e11077ec268c888133`.

## Gates

- `rch exec -- cargo bench -p ft-optim --bench optimizer_bench -- adamw/step_64x1024 --warm-up-time 1 --measurement-time 5 --sample-size 20` baseline passed.
- `rch exec -- cargo test -p ft-optim adamw_first_step_fused -- --nocapture` passed while the draft was applied: 2 tests.
- `rch exec -- cargo bench -p ft-optim --bench optimizer_bench -- adamw/step_64x1024 --warm-up-time 1 --measurement-time 5 --sample-size 20` after-run passed and regressed.
- Source lever reverted after the regression.
