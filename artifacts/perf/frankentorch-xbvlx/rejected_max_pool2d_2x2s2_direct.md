# frankentorch-xbvlx: reject max_pool2d 2x2 stride2 direct path

## Target

- Bead: `frankentorch-xbvlx`
- Label: `[perf] ft-kernel-cpu`
- Profile-backed row: `max_pool2d` over `[N,C,H,W]=[8,64,64,64]`, `2x2` kernel, stride `2`
- Candidate lever: add dtype-specialized `2x2`/stride-`2` direct forward branches for no-grad and grad-index collection while preserving the generic kernel's observable semantics.

## Baseline

Initial baseline on `vmi1152480`:

| Row | Median |
| --- | ---: |
| `max_pool2d/nograd` | `8.5073 ms` |
| `max_pool2d/nograd_f32` | `5.3922 ms` |
| `max_pool2d/grad` | `308.88 ms` |

Artifact: `pass1_baseline_max_pool2d.log`

## Isomorphism proof

The candidate's focused bit-identity proof passed while the source hunk existed:

- Command: `rch exec -- cargo test -j 1 -p ft-kernel-cpu max_pool2d_2x2s2_direct_matches_generic_bit_exact -- --nocapture`
- Result: passed
- Proof log SHA-256: `8484fce58d85c89c982f54cc51abba7ebe715de18436dc4f287c797384b9a56b`
- Artifact: `pass3_test_max_pool2d_direct_digest.log`

Behavior obligations checked by the focused proof:

- output shape and contiguous layout
- f64 and f32 output bit identity against the generic path
- first-tie max selection through strict `>` comparison
- NaN behavior induced by the same comparison order
- grad-index path preserving the generic linear input index choice
- no RNG, ordering, or floating-point arithmetic reassociation

`cargo check -j 1 -p ft-kernel-cpu --all-targets` passed with pre-existing example warnings (`pass4_check_ft_kernel_cpu.log`). `cargo clippy -j 1 -p ft-kernel-cpu --lib -- -D warnings` passed (`pass6_clippy_ft_kernel_cpu_lib.log`). The broader lib-test clippy command was blocked by pre-existing unrelated lint debt (`items_after_test_module`, `identity_op`) captured in `pass5_clippy_ft_kernel_cpu_lib_tests.log`.

## Same-worker control

Because the initial baseline and candidate landed on different workers, an unchanged baseline control was rerun from a detached baseline worktree on the same worker as the candidate (`ovh-a`).

| Row | Same-worker baseline | Candidate | Ratio |
| --- | ---: | ---: | ---: |
| `max_pool2d/nograd` | `1.7335 ms` | `2.1100 ms` | `0.8216x` |
| `max_pool2d/nograd_f32` | `893.47 us` | `1.9146 ms` | `0.4667x` |
| `max_pool2d/grad` | `73.325 ms` | `96.366 ms` | `0.7610x` |

Artifacts:

- Candidate: `pass7_candidate_max_pool2d.log`
- Baseline control: `pass8_baseline_max_pool2d_control.log`

## Verdict

Reject. The direct path regressed every same-worker benchmark row, so its score is `0.0` and below the `>=2.0` keep threshold. The candidate source hunk was removed before this closeout; only evidence and bead status remain.

Next route: avoid another hand-unrolled 2x2 pooling micro-lever unless a new profile isolates a different bottleneck. The deeper route is an output-tile or cache-layout primitive that reduces allocation/index overhead across pooling variants rather than adding another scalar direct branch.
