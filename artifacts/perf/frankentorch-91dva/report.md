# frankentorch-91dva SVD rank-deficient fast path

## Summary

Bead: `frankentorch-91dva`

Lever kept: guarded rank-revealing duplicate-column fast path for reduced square SVD. The fast path compresses duplicate input columns into a weighted unique-column factor, reuses the existing strict SVD on that smaller factor, lifts the decomposition, fills the zero-singular-value subspaces deterministically, and validates before returning. All non-triggering or failed-validation inputs fall back to the existing Golub-Reinsch path.

This deliberately avoids the rejected normal-equations family from `frankentorch-yyylo`.

## Same-worker benchmark

Worker: `ovh-a`

Command shape: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-a rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- 'svd_f64_256x256' --warm-up-time 1 --measurement-time 5 --sample-size 10`

| Row | Before median | After median | Delta |
| --- | ---: | ---: | ---: |
| `svd_f64_256x256` | `1.6296 s` | `140.06 ms` | `11.63x faster` |
| `svd_f64_256x256_wellcond` | `206.39 ms` | `210.90 ms` | `2.18% slower` |

The target row clears the keep gate. The well-conditioned row stays on the original SVD path; its cost is the duplicate-column guard prefilter and full fallback. The guard is restricted to square reduced SVD, finite input, `n >= 64`, and decisive duplicate-column rank loss.

Score: `Impact 5 * Confidence 5 / Effort 2 = 12.5`.

## Isomorphism proof

Ordering: duplicate-column groups are discovered by deterministic left-to-right column scan. Singular values come from the existing smaller strict SVD and are checked against `svdvals_contiguous_f64`; the result must be nonnegative and descending.

Tie and sign policy: the nonzero subspace inherits the existing inner SVD's deterministic sign choices. The nullspace is filled by deterministic monotonic basis completion for `U` and deterministic Helmert-style duplicate-column rows for `Vh`.

Floating-point policy: non-triggering cases fall back to the existing route. Triggering duplicate-column rank-deficient cases are numeric-contract equivalent rather than bit-identical, because they intentionally use a compressed factorization path. Runtime validation requires finite outputs, singular-value agreement, orthogonality, and reconstruction.

RNG: no RNG is introduced.

Fallback: if the fast path cannot prove the contract, it returns `None` and `svd_tall` continues through the original Golub-Reinsch implementation.

## Validation

Passed:

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-kernel-cpu svd -- --nocapture`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-kernel-cpu --all-targets --release`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`
- `git diff --check`
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`
- `ubs crates/ft-kernel-cpu/src/lib.rs`

Known gate limitation:

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo fmt -p ft-kernel-cpu --check` is refused by RCH because fmt is classified as a non-compilation command under remote-required mode.
- Local `cargo fmt -p ft-kernel-cpu --check` still fails on broad pre-existing `ft-kernel-cpu` formatter drift outside this SVD lever.

## Evidence files

- `baseline_clean_head_plus_eigfix_svd_256_ovha_final.log`
- `after_svd_256_sorted_prefilter_ovha_final.log`
- `test_svd_filter_sorted_prefilter.log`
- `cargo_check_ft_kernel_cpu_release_sorted_prefilter.log`
- `cargo_clippy_ft_kernel_cpu_sorted_prefilter.log`
- `golden_sha256_sorted_prefilter.log`
- `evidence_sha256_sorted_prefilter.txt`

## Next shifted target

Re-profile after landing. Current ready perf candidates are `frankentorch-l9xod` for non-symmetric eig and `frankentorch-3jmy3` for loss/autograd/tape work. The eig lane appears to have active peer edits in `ft-kernel-cpu`, so the next claim should re-check live bead state and reservations first.
