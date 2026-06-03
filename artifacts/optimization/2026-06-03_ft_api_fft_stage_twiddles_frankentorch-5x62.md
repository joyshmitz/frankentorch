# ft-api FFT Stage Twiddle Reuse - frankentorch-5x62

## Target

- Bead: `frankentorch-5x62`
- Crate: `ft-api`
- Benchmark: `fft_1d/262144pt`
- Command: `rch exec -- cargo bench -p ft-api --bench ops_bench -- fft_1d/262144pt --warm-up-time 1 --measurement-time 5 --sample-size 10`
- Worker: `ts2`

## Baseline And Profile

- Baseline: `[18.839 ms 19.095 ms 19.339 ms]`
- Profile-backed hotspot: the power-of-two `tensor_fft` path called `dft_inplace_1d`, which used the existing Cooley-Tukey implementation without a caller-provided stage table. That path rebuilds the per-stage twiddle slice as the FFT stage loop advances.
- Selected primitive: stage-twiddle table reuse for a radix-2 FFT pass.

## Lever

For power-of-two `fft_len`, `tensor_fft` now builds `fft_stage_twiddles(fft_len, false)` once and calls `dft_inplace_1d_with_stage_twiddles` with that table. Non-power-of-two inputs still use the existing fallback path.

## Result

- Candidate: `[18.261 ms 18.632 ms 18.967 ms]`
- p50 speedup: `19.095 ms / 18.632 ms = 1.0249x`
- Score: `Impact 1 * Confidence 2 / Effort 1 = 2.0`
- Verdict: keep.

## Isomorphism Proof

- Ordering preserved: yes. Bit-reversal order, stage order, in-stage `k` order, and output tensor order are unchanged.
- Tie-breaking unchanged: N/A. FFT has no ordering tie-breaker.
- Floating-point preserved: bit-identical for the proof fixture. The same twiddle formula feeds the same butterfly arithmetic in the same order; only the allocation/reuse point moves out of the stage loop.
- RNG unchanged: yes. No random state is read or modified.
- Shape/error behavior unchanged: yes. `n` handling, zero-padding/truncation, power-of-two dispatch, and non-power-of-two fallback are unchanged.
- Golden output: `artifacts/optimization/golden_outputs/ft_api_fft_stage_twiddles_frankentorch-5x62.txt`
- Golden sha256: `d177832396b4b83e559da3e5dde8db5060da766aad2b181b00e7666b0b8db85b`
- Fixture digest: `0x8216e8d606eb37d0`

## Validation

- Focused proof test: `rch exec -- cargo test -p ft-api tensor_fft_stage_twiddle_path_matches_per_stage_reference_bit_exact -- --nocapture` passed on `ts2`.
- Golden manifest: `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed.
- Crate check: `rch exec -- cargo check -p ft-api --all-targets` passed on `ts2`; existing warnings remain in `ft-kernel-cpu` and older `ft-api` tests.
- Diff hygiene: `git diff --check` passed.
- Rustfmt: `rch exec -- cargo fmt --package ft-api --check` exited 1 on existing formatting drift across older `ft-api` bench/test sections; this FFT hunk passed diff hygiene.
- UBS: `timeout 360s ubs crates/ft-api/src/lib.rs artifacts/optimization/golden_checksums.txt artifacts/optimization/golden_outputs/ft_api_fft_stage_twiddles_frankentorch-5x62.txt .beads/issues.jsonl` exited 1 after 284s on existing file-wide `ft-api` findings, including old unwrap/expect, test panic, indexing, and equality-pattern reports.
