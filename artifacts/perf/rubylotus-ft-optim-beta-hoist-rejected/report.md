# frankentorch-javw ft-optim Adam beta-complement hoist rejection

Agent: RubyLotus
Crate: `ft-optim`
Outcome: rejected, no source change kept

## Profile-backed target

After `frankentorch-ooza`, the clean optimizer Criterion profile on worker `ts1` showed AdamW as
the largest remaining `ft-optim` benchmark:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-optim --bench optimizer_bench -- --warm-up-time 1 --measurement-time 3 --sample-size 10

adamw/step_64x1024 [517.00 us 522.47 us 533.01 us]
adam/step_64x1024  [330.20 us 332.06 us 334.49 us]
sgd/step_64x1024   [190.45 us 193.53 us 197.14 us]
```

One attempted lever: hoist `1.0 - beta1` and `1.0 - beta2` once per parameter in Adam and AdamW
instead of spelling those invariant subtractions inside every element update.

## Rebenchmark

Same worker: `ts1`

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-optim --bench optimizer_bench -- --warm-up-time 1 --measurement-time 3 --sample-size 10

adamw/step_64x1024 [520.20 us 527.90 us 534.41 us]
adam/step_64x1024  [335.34 us 343.89 us 354.48 us]
sgd/step_64x1024   [196.18 us 198.45 us 201.57 us]
```

## Decision

Rejected. The same-worker signal regressed:

- AdamW median: 522.47 us -> 527.90 us
- Adam median: 332.06 us -> 343.89 us
- Score: below keep threshold because impact is negative

The source hunk was manually removed. This is a micro-lever rejection signal; the next pass should
attack a deeper primitive instead of another spelling-level AdamW tweak.

## Isomorphism note

The draft preserved behavior: it reused the exact same `1.0 - beta` subtraction results and left
optimizer ordering, floating-point formula order, state commit order, and RNG behavior unchanged.
The rejection is purely performance-based.
