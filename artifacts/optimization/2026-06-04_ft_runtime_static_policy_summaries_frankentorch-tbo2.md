# ft-runtime static policy evidence summaries

Bead: `frankentorch-tbo2`

## Target

`runtime_policy_evidence/new_and_switch_1024` creates 1024 `RuntimeContext`
values and switches each context once, producing two fixed policy evidence
entries per context. The old path assembled those fixed policy summaries from a
prefix and an execution-mode label on every policy record.

Alien primitive: compiled/static policy table for deterministic runtime
decisions, matching the graveyard guidance to consume compact deterministic
artifacts on hot paths instead of rebuilding fixed decision strings.

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-runtime --bench runtime_bench -- runtime_policy_evidence/new_and_switch_1024 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `vmi1153651`

Result:

```text
runtime_policy_evidence/new_and_switch_1024
time: [200.34 us 222.80 us 266.09 us]
```

## Change

Replace dynamic `policy_summary(prefix, mode)` assembly with two static
message-table helpers:

- `policy_initialized_summary(mode)`
- `policy_switched_summary(mode)`

The public ledger contract remains `String`-backed; `EvidenceLedger::record`
still owns the summary via `Into<String>`.

## Re-benchmark

Command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-runtime --bench runtime_bench -- runtime_policy_evidence/new_and_switch_1024 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `vmi1153651`

Result:

```text
runtime_policy_evidence/new_and_switch_1024
time: [176.62 us 181.02 us 187.57 us]
```

p50 speedup: `222.80 / 181.02 = 1.231x`

Score: `Impact 2 * Confidence 3 / Effort 1 = 6.0`

## Isomorphism

- Ordering preserved: yes. `RuntimeContext::new` still records the initial
  policy entry before any caller-visible work, and `set_mode` still records the
  switch entry after updating `mode`.
- Tie-breaking unchanged: not applicable.
- Floating point unchanged: not applicable.
- RNG unchanged: not applicable.
- Timestamp behavior unchanged: each policy record still calls
  `EvidenceLedger::record`, so timestamp capture remains per entry.
- Golden bytes unchanged: `ft_runtime_policy_pass20.txt` still matches exactly.

## Validation

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-runtime runtime_policy_evidence_golden_summary_matches_fixture -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-runtime --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-runtime --all-targets -- -D warnings
cargo fmt -p ft-runtime --check
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
git diff --check -- crates/ft-runtime/src/lib.rs artifacts/optimization/2026-06-04_ft_runtime_static_policy_summaries_frankentorch-tbo2.md .beads/issues.jsonl
ubs crates/ft-runtime/src/lib.rs artifacts/optimization/2026-06-04_ft_runtime_static_policy_summaries_frankentorch-tbo2.md .beads/issues.jsonl
```

All commands above passed.
