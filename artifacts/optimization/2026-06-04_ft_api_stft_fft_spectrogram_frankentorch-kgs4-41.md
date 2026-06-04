# ft-api STFT per-frame FFT - frankentorch-kgs4.41

## Target

- Bead: `frankentorch-kgs4.41`
- Crate: `ft-api`
- Benchmark: `stft/len32768_nfft512`
- Lever: compute each power-of-two STFT frame with the existing radix-2 FFT/stage-twiddle path, then transpose frame-major output back to the existing bin-major `[freq_bins, frames]` layout.

## Profile-backed baseline

Clean `HEAD` baseline from an extracted copy under `/data/projects/.scratch`, via `rch` on worker `vmi1149989`:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-api --bench ops_bench -- stft/len32768_nfft512 --warm-up-time 1 --measurement-time 5 --sample-size 20
stft/len32768_nfft512   time: [8.2820 ms 8.5425 ms 8.8311 ms]
```

The remaining hotspot after the dense-DFT twiddle precompute was the `O(freq_bins * n_fft)` DFT for every frame. For `n_fft=512`, the transform size is power-of-two, so the existing safe-Rust radix-2 FFT path applies directly.

## Result

Same benchmark, same worker, current candidate:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-api --bench ops_bench -- stft/len32768_nfft512 --warm-up-time 1 --measurement-time 5 --sample-size 20
stft/len32768_nfft512   time: [1.7817 ms 2.4087 ms 2.9633 ms]
```

P50 speedup:

```text
8.5425 ms / 2.4087 ms = 3.546x
```

Score: `Impact 4 * Confidence 4 / Effort 2 = 8.0`; keep.

## Isomorphism proof

- Ordering preserved: yes. Results are transposed back to the same bin-major `[freq_bins, frames]` order.
- Tie-breaking unchanged: N/A. STFT has no comparison or tie rule.
- Floating-point: intentional FFT arithmetic change for power-of-two `n_fft`; verified against the dense DFT reference with `1e-7 * (1 + abs(reference))` tolerance. Non-power-of-two `n_fft` keeps the dense fallback.
- RNG unchanged: yes. No random state is read or modified.
- Shape/error/ledger behavior unchanged: yes. Validation, dtype selection, windowing, scaling, output shape, and evidence ledger text are unchanged.
- Golden output: `artifacts/optimization/golden_outputs/ft_api_stft_fft_spectrogram_frankentorch-kgs4-41.txt`

## Validation

Passed:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-api stft_matches_dense_dft_within_tolerance -- --nocapture
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-api --all-targets
```

Additional gate notes:

```text
Initial dirty-worktree clippy reached an unrelated ft-dispatch unused import from
another in-flight hash-index registry change before it reached ft-api. That
surface was later landed separately as c108da9a.

cargo fmt --package ft-api --check reported broad pre-existing ft-api formatting
drift outside this STFT lever, so the closeout uses the focused test, checksum,
crate check, and artifact-local diff checks rather than rewriting unrelated
formatting.
```
