# ft-dispatch Schema Name Reuse Rejection

- Bead: `frankentorch-g7t2`
- Crate: `ft-dispatch`
- Target benchmark: `schema_registry/register_1024`
- Skills: `/extreme-software-optimization`, `/alien-graveyard`, `/alien-artifact-coding`

## Profile Target

`SchemaRegistry::register` normalizes schema names, validates dispatch keysets,
detects duplicates through `BTreeMap::entry`, maps the base operator, and stores
the dispatch entry. The benchmark registers 1024 unique schema names.

Baseline:

```text
worker: vmi1149989
command: rch exec -- cargo bench -p ft-dispatch --bench dispatch_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
schema_registry/register_1024: [294.47 us 310.57 us 329.10 us]
```

Rejected candidate:

```text
lever: compute OpSchemaName::unambiguous_name once in SchemaRegistry::register and reuse it for digest64 plus BTreeMap insertion
worker: vmi1153651
command: rch exec -- cargo bench -p ft-dispatch --bench dispatch_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
schema_registry/register_1024: [533.51 us 552.46 us 574.44 us]
```

The candidate did not prove a win. The source hunk was manually reverted; no
runtime optimization shipped.

## Alien Recommendation Card

Candidate primitive: constants-kill-you allocation elimination / single
materialization of normalized schema names.

Expected value after measurement: Impact 0 x Confidence 4 / Effort 1 = 0.

Fallback: keep the current register implementation. The duplicate allocation is
not worth changing without same-worker benchmark evidence that beats the current
path.

## Isomorphism Proof

The rejected candidate was intended to preserve behavior:

- Ordering: `BTreeMap` remained the registry storage, so iteration order stayed
  sorted by normalized name.
- Tie-breaking: duplicate detection still used `BTreeMap::entry` and the same
  occupied-key string.
- Floating point: no floating-point operations are involved.
- RNG: schema registration uses no RNG.
- Errors: keyset validation remained before name normalization; duplicate
  detection remained before unsupported-operator detection.
- Golden output: existing schema registry fixture remains pinned at sha256
  `1ed775b109f682bad9b11c04e7818845bd57927ad48ef497c7ee2d6b7049207d`.

## Result

Rejected. The committed diff for this bead is the evidence artifact plus the
`.beads` closeout; no `ft-dispatch` source change remains.
