# frankentorch-8x2i recurrent call amortization

## Pass 1 baseline/profile

Command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-api --bench ops_bench -- recurrent_forward --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts1`

Criterion output:

- `recurrent_forward/lstm_seq64_batch1_128x128`: `[3.5421 ms 3.6060 ms 3.6886 ms]`
- `recurrent_forward/gru_seq64_batch1_128x128`: `[2.5523 ms 2.5884 ms 2.6339 ms]`
- `recurrent_forward/rnn_tanh_seq64_batch1_128x128`: `[857.73 us 873.46 us 891.54 us]`

Profile-backed residual from predecessor bead `frankentorch-3k4v`: raw batch=1 LSTM/GRU/RNN still invokes one tiny RHS-transposed recurrent multiply per timestep after the input projection is already batched across the sequence. The rejected predecessor materialized `W_hh^T` layout lever was not call-amortization and regressed LSTM/GRU.

## Alien primitive and proof plan

Primitive: cache-resident row-vector microkernel for `m == 1` RHS-transposed f64 multiply, harvested from communication-avoiding kernel guidance: keep hot recurrent state resident, avoid tiny BLAS call setup, and preserve the existing per-output `k` accumulation contract.

One lever only: specialize `matmul_rhs_transposed_contiguous_f64_into` for `m == 1`.

Isomorphism obligations:

- API/ABI preserved: same helper signature, same error checks, same output length truncation/growth behavior.
- Ordering preserved: timestep order, layer order, gate order, and output flattening remain owned by `ft-api` recurrent code and are untouched.
- Tie-breaking unchanged: no branches choose among equal values.
- Floating-point drift gate: candidate is accepted only if the row-vector helper matches allocating `dgemm_bt` bit-for-bit for recurrent shapes and recurrent golden checksums remain unchanged.
- RNG unchanged: recurrent raw-forward paths are deterministic and use no RNG.
- Resource envelope: one fewer `matrixmultiply` call path per recurrent timestep for batch=1; no new allocations on the hot helper path.
- Golden outputs: run `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`.

Opportunity score before implementation: Impact `3.0` x Confidence `0.75` / Effort `1.0` = `2.25`. The confidence discount is for the strict bitwise FP gate against `matrixmultiply`.

## Pass 4-6 candidate result

Candidate command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-kernel-cpu matmul_rhs_transposed_row_vector_into_matches_allocating_bit_exact -- --nocapture
```

Worker: `ts1`

Result: rejected before benchmarking. The focused bitwise helper proof failed on the first recurrent shape:

```text
row-vector matmul diverged at 0 for (1,128,512): got 1.424984613091199 vs want 1.4249846130912007
```

Verdict: source candidate removed. No rebench was run because floating-point parity failed and the recurrent golden checksum gate would not be meaningful after a known low-bit drift.

Score: `0.0` because behavior equivalence failed before the performance gate.

Next primitive: a matrixmultiply-equivalent recurrent panel/gate kernel whose traversal is derived from the current `dgemm_bt` accumulation order instead of a scalar dot-order rewrite. The target ratio remains `>=2.0` Score by amortizing call/setup overhead while preserving exact FP bits.
