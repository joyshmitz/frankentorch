# frankentorch-biqbj - recurrent prechecked-matmul rejection

## Target

- Bead: `frankentorch-biqbj`
- Surface: `ft-api` recurrent raw forward (`tensor_lstm`, `tensor_gru`, `tensor_rnn`)
- Baseline worker: `vmi1152480`
- Candidate worker: `vmi1152480`

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -j 1 -p ft-api --bench ops_bench recurrent_forward -- --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Rows:

| Case | Baseline median |
| --- | ---: |
| `recurrent_forward/lstm_seq64_batch1_128x128` | `3.8642 ms` |
| `recurrent_forward/gru_seq64_batch1_128x128` | `2.6480 ms` |
| `recurrent_forward/rnn_tanh_seq64_batch1_128x128` | `953.25 us` |

## Candidate

One lever: call the existing prechecked `dgemm_bt` helper from each recurrent
step and keep exact-size persistent scratch buffers.

This preserved the existing `matrixmultiply` traversal. It only attempted to
remove recurrent-step checked-mul/resize branches.

## Behavior proof

Command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -j 1 -p ft-api raw_forward_golden_isomorphism -- --nocapture
```

Result on `vmi1152480`: passed 3/3.

Isomorphism ledger:

- ordering: layer, direction, timestep, batch, gate, and output order unchanged;
- tie-breaking: no comparison/tie path changed;
- floating point: recurrent projection still delegated to `dgemm_bt`; gate equations and bias add order unchanged;
- RNG: recurrent raw forward uses no RNG;
- golden tests: `lstm_raw_forward_golden_isomorphism`, `gru_raw_forward_golden_isomorphism`, and `rnn_raw_forward_golden_isomorphism` passed.

## Rebench

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKERS=vmi1152480 rch exec -- cargo bench -j 1 -p ft-api --bench ops_bench recurrent_forward -- --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Rows:

| Case | Baseline median | Candidate median | Ratio |
| --- | ---: | ---: | ---: |
| `lstm_seq64_batch1_128x128` | `3.8642 ms` | `3.9855 ms` | `0.970x` |
| `gru_seq64_batch1_128x128` | `2.6480 ms` | `2.8170 ms` | `0.940x` |
| `rnn_tanh_seq64_batch1_128x128` | `953.25 us` | `1.0902 ms` | `0.874x` |

Score: `0.0`. The candidate is proof-clean but regresses every acceptance row.

## Graveyard/artifact mapping

- No-gaps directive: pure safe-Rust BLAS-class kernels only; no C BLAS/LAPACK/MKL/XLA linkage.
- Alien-graveyard match: dense-kernel locality and constants-aware optimization (`§6.5` polyhedral/locality, `§9.7` communication-avoiding dense linear algebra, `§16.7` constants kill you).
- Alien-artifact family: numerical linear algebra/matrix methods; proof obligations are bitwise FP preservation and recurrent-state ordering.

## Verdict

Rejected. Source hunk removed.

Do not repeat these recurrent families without a new measured reason:

- materialized `W_hh^T`;
- persistent packed RHS replay through private matrixmultiply internals;
- scalar row-vector rewrite;
- branch/bias splitting;
- borrowed recurrent storage;
- flat sequence/output workspace;
- prechecked recurrent scratch.

Next route: file a new profile-backed perf child from current evidence and attack a different hotspot surface, or return to recurrent only if a new profile isolates a cost component outside the rejected families above.
