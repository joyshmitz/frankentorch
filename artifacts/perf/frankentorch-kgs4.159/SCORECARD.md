# frankentorch-kgs4.159 — parallel first-dim (stride_outer==1) FFT pass — completes n-D FFT parallelization

Date: 2026-06-21
Agent: cc
Follow-up to kgs4.158 (which covered stride_outer>=2 non-last dims). Together they make the
whole n-D FFT family fully parallel across every dim pass.

## Lever

After kgs4.158, the FIRST transform dim of an n-D tensor (stride_outer==1, e.g. the common
2-D-FFT row pass) was STILL serial: its `stride_inner` lanes are STRIDED columns
(k*stride_inner+inner), not a contiguous chunk, so `par_chunks_mut` can't apply. New branch:
parallel-map each lane to a transformed (re,im) buffer (immutable strided gather +
compute-bound dft), collect in lane order, then scatter back serially. The parallel part is
the dft compute; the serial scatter is a cheap O(total) bandwidth tail.

## Correctness (bit-exact)

Deterministic per-lane dft, identical gather/scatter indices, lane-ordered collect → bit-for-bit
identical to the serial loop. Same-build A/B reported IDENTICAL checksum (`7.996144e6`). ft-api
`--lib fft` 31 passed / 0 failed; ft-conformance green.

## Measurement (same-host, same-build A/B, no-grad fftn, 32 threads, example fftn_strided_ab)

| Shape | dim-0 serial (baseline) | dim-0 parallel (this) | internal |
| --- | ---: | ---: | --- |
| `[48,128,128]` (dim-0 bottleneck: 16384 strided lanes of O(48²) DFT) | `567–569 ms` | `54.9–55.7 ms` | **~10.3x faster** |

## Verdict: KEEP — internal ~10.3x, PyTorch loss

vs PyTorch fftn: PyTorch ~0.6 ms (pocketfft mixed-radix even for 48=16×3) vs FT ~55 ms; FT
remains ~90x slower because FT uses an O(N²) DFT for non-power-of-2 sizes — an algorithmic gap,
separate from this parallel pass. Internal core-scaling win (the FFT-vein disposition), bit-exact,
removes the last serial n-D FFT bottleneck. With kgs4.158, every dim pass of fftn/ifftn/rfftn/
irfftn is now parallel.

## Win/loss/neutral vs PyTorch (32t): `0W / 1L / 0N` (internal ~10.3x kept)

## Gates
- `cargo test -p ft-api --release --lib fft`: 31 passed, 0 failed.
- `cargo test -p ft-conformance --release`: green.
