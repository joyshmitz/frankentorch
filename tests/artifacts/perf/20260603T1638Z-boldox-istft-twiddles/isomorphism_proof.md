# frankentorch-ft-api-istft-twiddle-verz ISTFT Twiddle Precompute

## Change

`istft_reconstruct_f64` now precomputes each inverse-DFT twiddle
`(cos(angle), sin(angle))` for every `(n, k)` pair once per call when
`n_fft * n_fft <= 1 << 20`, then reuses the table across all frames. Larger
transforms fall back to the previous inline `cos/sin` path, so extreme input
sizes do not gain a new allocation/panic surface.

This remains the existing direct inverse DFT. This pass deliberately does not
substitute a Cooley-Tukey FFT because bit-level floating-point parity is
required for this lever.

## Proof Obligations

- Ordering preserved: yes. Per frame, output sample `n` is still produced in
  ascending order. For each sample, the summation over full-spectrum index `k`
  is still ascending.
- Tie-breaking: N/A. ISTFT has no ordering/tie branch.
- Floating-point: preserved for observable outputs. The twiddle table computes
  the same `angle` expression and same `cos`/`sin` values once for each `(n, k)`;
  each sample still accumulates `coeff * twiddle` in ascending `k` order and
  applies the same scale.
- RNG seeds: N/A. ISTFT has no RNG path beyond the external benchmark fixture.
- DType/shape: unchanged for Complex64->F32 and Complex128->F64 paths.
- Overlap-add: unchanged. Frame reconstruction may run in parallel as before,
  but overlap-add into `signal`/`envelope` remains serial in ascending frame
  order.
- Ledger text: unchanged.
- Fallback: for `n_fft * n_fft > 1 << 20` or multiplication overflow, the old
  inline twiddle calculation path is used.

## Validation

- `rch exec -- cargo test -p ft-api istft_parallel_match_serial_bit_exact -- --nocapture`
  passed on `vmi1149989`.
- Golden fixture file: `golden_istft_outputs.txt`.
- Same-worker remote baseline: `istft/nfft512_frames256`
  `[200.55 ms 220.72 ms 243.49 ms]` on `vmi1293453`.
- Same-worker remote after: `istft/nfft512_frames256`
  `[15.689 ms 16.698 ms 17.959 ms]` on `vmi1293453`.

## Score

Impact 5 x confidence 4 / effort 2 = 10.0. Keep.
