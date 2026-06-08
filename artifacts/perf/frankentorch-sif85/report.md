# frankentorch-sif85 - recurrent gate-panel candidate rejection

## Target

- Bead: `frankentorch-sif85`
- Title: `[perf][no-gaps] ft-api recurrent gate-panel kernel with matrixmultiply-order proof`
- Crate: `ft-api`
- Benchmark group: `recurrent_forward`
- Worker discipline: same-worker `rch` on `fmd`
- Baseline source: clean detached worktree at `28b0f1ac`

## Profile-backed baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKERS=fmd rch exec -- \
  cargo bench -p ft-api --bench ops_bench -- recurrent_forward \
  --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Baseline medians on `fmd`:

| Case | Baseline |
| --- | ---: |
| `lstm_seq64_batch1_128x128` | `3.2965 ms` |
| `gru_seq64_batch1_128x128` | `2.3169 ms` |
| `rnn_tanh_seq64_batch1_128x128` | `708.22 us` |

## Candidate

One lever was tested: split recurrent LSTM/GRU/RNN gate loops into bias-present fast paths so the benchmark's present-bias case does not re-check `Option` state through tiny closures in every gate lookup.

No matrixmultiply call, timestep ordering, gate ordering, hidden/cell layout, output flattening, RNG behavior, or dispatcher surface was changed.

## Behavior proof while candidate was present

Focused recurrent golden isomorphism tests:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKERS=fmd rch exec -- \
  cargo test -p ft-api raw_forward_golden_isomorphism -- --nocapture
```

Result: passed 3/3 on `fmd`.

- `lstm_raw_forward_golden_isomorphism`: ok
- `gru_raw_forward_golden_isomorphism`: ok
- `rnn_raw_forward_golden_isomorphism`: ok

Golden artifact checksum:

```bash
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
```

Result: all locally present golden artifacts passed.

Crate-scoped compile gate:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKERS=fmd rch exec -- cargo check -p ft-api --all-targets
```

Result: passed on `fmd`.

Isomorphism ledger:

- Ordering: layer, direction, timestep, batch, and gate iteration order unchanged.
- Tie-breaking: no ordering/tie code touched.
- Floating point: candidate preserved per-gate grouping for the tested bias-present paths:
  - LSTM/RNN: `xw + hg`, then `+= b_ih`, then `+= b_hh`.
  - GRU: existing `gi_all` already includes `b_ih`; recurrent `gh + b_hh` was combined at the same gate expression site.
- RNG: no RNG paths involved or changed.
- Final source state: rejected source hunk removed; `crates/ft-api/src/lib.rs` has no retained diff from this candidate.

## Same-worker rebench

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKERS=fmd rch exec -- \
  cargo bench -p ft-api --bench ops_bench -- recurrent_forward \
  --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Candidate medians on `fmd`:

| Case | Baseline | Candidate | Ratio |
| --- | ---: | ---: | ---: |
| `lstm_seq64_batch1_128x128` | `3.2965 ms` | `3.1666 ms` | `1.041x` |
| `gru_seq64_batch1_128x128` | `2.3169 ms` | `2.2986 ms` | `1.008x` |
| `rnn_tanh_seq64_batch1_128x128` | `708.22 us` | `718.23 us` | `0.986x` |

Score:

```text
Impact 1.01 x Confidence 0.95 / Effort 0.60 = 1.60
```

The lever does not meet the `Score >= 2.0` keep gate and regresses the RNN tanh case, so it was rejected.

## Verdict

Rejected. The branch-split micro-lever is the wrong family for this profile target. The next bead should attack the deeper primitive named by the original target: a matrixmultiply-order recurrent panel/gate kernel derived from the existing `dgemm_bt` traversal, with a bit-exact helper proof before Criterion.
