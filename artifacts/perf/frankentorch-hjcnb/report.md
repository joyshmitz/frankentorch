# frankentorch-hjcnb - matrixmultiply-order recurrent panel kernel

## Target

- Bead: `frankentorch-hjcnb`
- Title: `[perf][no-gaps] ft-api matrixmultiply-order recurrent panel kernel`
- Crate surface: `ft-api`, likely `ft-kernel-cpu`
- Benchmark group: `recurrent_forward`
- Worker discipline: same-worker `rch`
- Baseline source: clean detached worktree at `d0791d23`

## Pass 1 - profile-backed baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKERS=fmd rch exec -- \
  cargo bench -p ft-api --bench ops_bench -- recurrent_forward \
  --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Baseline medians on `fmd`:

| Case | Baseline |
| --- | ---: |
| `lstm_seq64_batch1_128x128` | `3.2038 ms` |
| `gru_seq64_batch1_128x128` | `2.3329 ms` |
| `rnn_tanh_seq64_batch1_128x128` | `742.04 us` |

Profile-backed residual:

- Batch=1 recurrent raw-forward still invokes one tiny RHS-transposed recurrent multiply per timestep.
- Branch splitting is excluded by `frankentorch-sif85`: it scored `1.60` and regressed RNN.
- Scalar row-vector dot rewrites are excluded by `frankentorch-8x2i`: they failed bitwise matrixmultiply parity before benchmarking.

## Candidate constraints

The next lever must be one matrixmultiply-order primitive, not a gate-loop micro-tune:

- preserve existing `dgemm_bt` accumulation bits or prove a helper is bit-exact against it before recurrent rebench;
- preserve layer, direction, timestep, batch, gate, hidden/cell, and output ordering;
- preserve FP grouping around input projection, recurrent projection, bias addition, nonlinearities, and final tensor flattening;
- preserve RNG absence;
- keep source only if same-worker Score is `>= 2.0`.

## Pass 2-6 - padded-M matrixmultiply panel rejection

Candidate:

- Add a batch-1 RHS-transposed f64 helper that pads the single recurrent row to eight rows before calling the same `dgemm_bt`.
- Wire LSTM/GRU/RNN batch-1 recurrent projections through that helper.
- Rationale: matrixmultiply's x86 f64 kernels use eight-row panels; `m == 1` goes through masked edge handling. Padding to the panel height keeps row 0 inside matrixmultiply's own traversal and tests whether avoiding the masked-M output path beats the extra scratch/copy work.

Behavior proof while candidate was present:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKERS=fmd rch exec -- \
  cargo test -p ft-kernel-cpu \
  matmul_rhs_transposed_m1_padded8_matches_dgemm_bt_bit_exact -- --nocapture
```

Result: passed on `fmd`.

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKERS=fmd rch exec -- \
  cargo test -p ft-api raw_forward_golden_isomorphism -- --nocapture
```

Result: passed 3/3 on `fmd`.

- `lstm_raw_forward_golden_isomorphism`: ok
- `gru_raw_forward_golden_isomorphism`: ok
- `rnn_raw_forward_golden_isomorphism`: ok

Golden SHA-256:

```bash
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
```

Result: all locally present golden artifacts passed.

Compile gate:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKERS=fmd rch exec -- cargo check -p ft-api --all-targets
```

Result: passed on `fmd`.

Same-worker rebench:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKERS=fmd rch exec -- \
  cargo bench -p ft-api --bench ops_bench -- recurrent_forward \
  --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Candidate medians on `fmd`:

| Case | Baseline | Candidate | Ratio |
| --- | ---: | ---: | ---: |
| `lstm_seq64_batch1_128x128` | `3.2038 ms` | `3.1829 ms` | `1.007x` |
| `gru_seq64_batch1_128x128` | `2.3329 ms` | `2.2998 ms` | `1.014x` |
| `rnn_tanh_seq64_batch1_128x128` | `742.04 us` | `721.66 us` | `1.028x` |

Score:

```text
Impact 1.02 x Confidence 0.95 / Effort 0.90 = 1.08
```

Verdict: rejected. The proof succeeded, but the win is below the `Score >= 2.0` gate. Source hunk removed; no `ft-api` or `ft-kernel-cpu` optimization code retained.

Next primitive: persistent prepacked recurrent RHS panels. The measured residual is not masked-M edge overhead; the deeper target is the repeated packing/setup of the same `W_hh` rows across 64 timesteps. The next bead should attack reusable matrixmultiply-order packed RHS panels with a bit-exact helper proof before recurrent Criterion.
