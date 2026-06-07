# frankentorch-3k4v recurrent pretranspose probe

## Scope

- Bead: `frankentorch-3k4v`
- Target: no-grad raw recurrent forward residual for LSTM/GRU/RNN.
- Profile-backed symptom: after the raw recurrent rewrite, batch=1 still pays one
  recurrent `dgemm_bt` call per timestep. The residual is dominated by fixed
  matrixmultiply call/layout overhead rather than session/tape construction.
- One lever tested: materialize `W_hh^T` once per layer and call a contiguous
  `m x k` by `k x n` GEMM helper for each recurrent step.

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-api --bench ops_bench -- recurrent_forward --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts1`

Criterion intervals:

| Benchmark | Baseline |
| --- | ---: |
| `recurrent_forward/lstm_seq64_batch1_128x128` | `[3.5017 ms 3.6064 ms 3.7467 ms]` |
| `recurrent_forward/gru_seq64_batch1_128x128` | `[2.6710 ms 2.8719 ms 3.0523 ms]` |
| `recurrent_forward/rnn_tanh_seq64_batch1_128x128` | `[1.1203 ms 1.1268 ms 1.1336 ms]` |

## Candidate Selection

Alien-graveyard route: cache locality, morsel/vectorized execution, and
batching-to-amortize-overhead ideas point to eliminating strided RHS access and
making the recurrent matrix multiply layout friendlier without changing the
gate equations or hidden-state loop order.

Pre-score:

| Candidate | Impact | Confidence | Effort | Score |
| --- | ---: | ---: | ---: | ---: |
| Materialize recurrent `W_hh^T` once and use contiguous GEMM per step | 2.0 | 3.0 | 2.0 | 3.0 |

Proof obligation: the candidate may only change the RHS memory layout passed to
the same matrixmultiply accumulation primitive. It must preserve timestep order,
layer/direction order, gate order, bias addition order, hidden/cell state layout,
and no-grad/tape behavior.

## Candidate Proof

Commands run while the candidate was present:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-kernel-cpu matmul_contiguous_f64_into_reuses_buffer_bit_exact -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-api raw_forward_golden_isomorphism -- --nocapture
```

Results:

- `ft-kernel-cpu` focused helper test passed on `vmi1293453`.
- `ft-api` raw LSTM/GRU/RNN golden isomorphism tests passed on `ts1`.

Isomorphism ledger:

- Ordering preserved: candidate preserved timestep, layer, direction, batch, and gate order.
- Tie-breaking unchanged: N/A; no comparisons or top-k ordering.
- Floating-point: candidate used the same matrixmultiply primitive after explicit materialization; raw golden tests passed. The kept tree removes the candidate, so production FP order is unchanged from baseline.
- RNG seeds: unchanged; benchmark data uses deterministic trigonometric values and production code uses no RNG here.
- Golden outputs: focused recurrent golden tests passed for the candidate; kept tree restores the exact baseline source path.

## Same-Worker Rebench

Command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-api --bench ops_bench -- recurrent_forward --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts1`

Criterion intervals:

| Benchmark | Baseline | Candidate | Ratio |
| --- | ---: | ---: | ---: |
| `recurrent_forward/lstm_seq64_batch1_128x128` | `3.6064 ms` | `7.8714 ms` | `0.46x` |
| `recurrent_forward/gru_seq64_batch1_128x128` | `2.8719 ms` | `3.4168 ms` | `0.84x` |
| `recurrent_forward/rnn_tanh_seq64_batch1_128x128` | `1.1268 ms` | `1.0579 ms` | `1.07x` |

Post-score:

| Candidate | Impact | Confidence | Effort | Score |
| --- | ---: | ---: | ---: | ---: |
| Materialized recurrent `W_hh^T` contiguous GEMM | 0.46 | 0.95 | 1.0 | 0.44 |

## Verdict

Rejected. LSTM and GRU regressions dominate the small RNN win, so the lever is
below the Score >= 2.0 gate. The candidate source and helper were removed.

Kept artifact: the `recurrent_forward` Criterion benchmark group remains because
it is profiling infrastructure for the recurrent residual and does not affect
runtime behavior.

Next no-gaps route: avoid another layout-only micro-lever. The next bead should
target a materially different primitive: BLAS-order-preserving recurrent
call-amortization, such as a matrixmultiply-equivalent batched/panel kernel or a
raw recurrent sweep that preserves the proven accumulation order while reducing
per-timestep invocation overhead.

## Kept-State Verification

Commands:

```bash
git diff --check
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-api raw_forward_golden_isomorphism -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-api --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-api --all-targets -- -D warnings
cargo fmt -p ft-api --check
timeout 120s ubs crates/ft-api/benches/ops_bench.rs crates/ft-api/src/lib.rs
```

Results:

- `git diff --check`: passed.
- Golden SHA-256 check: passed for all locally present artifacts.
- Focused recurrent golden tests: passed 3 tests on `ts1`.
- `cargo check -p ft-api --all-targets`: passed on `ts1`.
- `cargo clippy -p ft-api --all-targets -- -D warnings`: failed on existing
  broad `ft-api` lint debt, ending with 210 errors across unrelated code
  (`needless_range_loop`, `manual_is_multiple_of`, `manual_div_ceil`,
  `excessive_precision`, `too_many_arguments`, and similar). The recurrent
  `unused_mut` warnings observed during the first proof run were removed before
  the final check.
- `cargo fmt -p ft-api --check`: failed on existing broad `ft-api` rustfmt drift
  across multiple benches/examples and `src/lib.rs`; no whole-crate formatting
  churn was applied in this performance pass.
- UBS: timed out after 120 seconds without producing findings.
