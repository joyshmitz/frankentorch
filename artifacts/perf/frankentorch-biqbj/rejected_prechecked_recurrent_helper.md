# frankentorch-biqbj - prechecked recurrent helper rejection

## Candidate

One rejected lever replaced the recurrent `matmul_rhs_transposed_contiguous_f64_into`
calls in LSTM/GRU/RNN raw forward with a prechecked exact-size scratch helper.
The helper still delegated arithmetic to `gemm::dgemm_bt`; it only removed the
checked multiply / resize branch from each recurrent timestep.

## Behavior proof

`rch exec -- cargo test -j 1 -p ft-api raw_forward_golden_isomorphism -- --nocapture`
on `vmi1152480` passed:

- `lstm_raw_forward_golden_isomorphism`
- `gru_raw_forward_golden_isomorphism`
- `rnn_raw_forward_golden_isomorphism`

Ordering, tie behavior, RNG behavior, gate order, and `dgemm_bt` floating-point
traversal were unchanged while the candidate was present.

## Same-worker benchmark gate

Baseline on `vmi1152480`:

- LSTM: `[3.7295 ms 3.8642 ms 4.0976 ms]`
- GRU: `[2.5497 ms 2.6480 ms 2.8022 ms]`
- RNN tanh: `[914.61 us 953.25 us 991.98 us]`

Candidate on `vmi1152480`:

- LSTM: `[3.8357 ms 3.9855 ms 4.2086 ms]`
- GRU: `[2.6855 ms 2.8170 ms 2.9286 ms]`
- RNN tanh: `[1.0101 ms 1.0902 ms 1.1838 ms]`

Median ratios:

- LSTM: `0.970x`
- GRU: `0.940x`
- RNN tanh: `0.874x`

Score: `0.0`. The candidate regressed all measured rows and was removed.

## Route

Do not repeat checked-branch, resize, scalar row-vector, bias-branch, packed-RHS
replay, borrowed-storage, or flat-workspace families for this lane. The next
attempt needs a materially different recurrent primitive derived from the
existing matrixmultiply traversal or a different profile-backed perf bead.
