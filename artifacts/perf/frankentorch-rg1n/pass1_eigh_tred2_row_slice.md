# frankentorch-rg1n pass 1: row-sliced Householder tridiagonal reduction

Target: `ft-kernel-cpu` full-vector `eigh_f64_256x256`, shared Householder tridiagonalization in `eigh_tred2_reduce`.

Profile and baseline:

- Prior profile in `frankentorch-rg1n`: after vector-layout passes, `eigh_f64_256x256` median `11.925 ms`, `eigvalsh_f64_256x256` median `7.964 ms`.
- Fresh same-worker baseline on `ts1`:
  - `eigh_f64_256x256`: `[10.579 ms 10.725 ms 10.897 ms]`
  - `eigvalsh_f64_256x256`: `[8.1234 ms 8.3753 ms 8.6995 ms]`
- The split leaves the shared reduction/QL path as the larger residual, with full-vector-only work around `2.35 ms`.

Alien primitive selection:

- Canonical graveyard entry 9.6 points to communication-avoiding dense linear algebra and BLAS-3 inner kernels as the deeper direction.
- Family 34 numerical linear algebra proof requirements demand exact decomposition behavior, orthogonality/reconstruction checks, and conditioning/error ledgers.
- A full blocked `dsytrd` or tridiagonal D&C/secular merge would reassociate floating-point work, so this pass took the exact-arithmetic sublever first: keep the EISPACK operation stream, but make the active Householder row explicit as a slice.

Lever:

- In `eigh_tred2_reduce`, split `z` at the active row once per Householder step.
- Read/write the active reflector row through `row_i`, and previous rows through `previous_rows`.
- Preserve every `j` loop, `k` loop, dot-product order, rank-2 update expression, and `d`/`e` update.

Behavior proof:

- Ordering and tie-breaking: unchanged. `eigh_contiguous_f64` still sorts `(value, old_col)` pairs with `f64::total_cmp`, and `eigvalsh_contiguous_f64` still sorts `d` with `total_cmp`.
- Floating point: bit-identical. The Householder scale loop, normalization loop, `gg` dot-product loops, `f` accumulation, `hh`, and trailing update use the same expressions in the same order. Only address calculation moved from repeated `z[i * n + k]` indexing to slices over the same storage.
- RNG: not used.
- Golden before SHA-256: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- Golden after SHA-256: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- `cmp artifacts/perf/frankentorch-rg1n/eigh_golden_before.txt artifacts/perf/frankentorch-rg1n/eigh_golden_after.txt` passed.

After benchmark:

- Worker: `ts1`
- `eigh_f64_256x256`: `[10.175 ms 10.253 ms 10.352 ms]`
- `eigvalsh_f64_256x256`: `[7.5509 ms 7.8526 ms 8.1005 ms]`
- Full `eigh` median speedup: `10.725 / 10.253 = 1.046x`.
- Values-only median speedup: `8.3753 / 7.8526 = 1.067x`.
- Score: `Impact 2.0 x Confidence 4.0 / Effort 1.5 = 5.3`, keep.

Verification:

- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo check -p ft-kernel-cpu --all-targets`: passed.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-kernel-cpu eigh -- --nocapture`: passed, 6 tests.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`: passed.
- `git diff --check -- crates/ft-kernel-cpu/src/lib.rs artifacts/perf/frankentorch-rg1n/pass1_eigh_tred2_row_slice.md artifacts/perf/frankentorch-rg1n/eigh_golden_before.txt artifacts/perf/frankentorch-rg1n/eigh_golden_after.txt artifacts/optimization/golden_checksums.txt .skill-loop-progress.md .beads/issues.jsonl`: passed.
- `ubs crates/ft-kernel-cpu/src/lib.rs artifacts/perf/frankentorch-rg1n/pass1_eigh_tred2_row_slice.md artifacts/perf/frankentorch-rg1n/eigh_golden_before.txt artifacts/perf/frankentorch-rg1n/eigh_golden_after.txt artifacts/optimization/golden_checksums.txt .skill-loop-progress.md .beads/issues.jsonl`: exit 0, 0 critical findings. Broad warning inventory in `ft-kernel-cpu/src/lib.rs` remains outside this lever.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`: blocked by unrelated peer-owned Winograd bench surface (`crates/ft-kernel-cpu/benches/winograd_bench.rs`: unused imports and unresolved private `gemm`).
- `cargo fmt -p ft-kernel-cpu --check`: blocked by pre-existing formatting drift in `linalg_bench.rs` and broad `ft-kernel-cpu/src/lib.rs` sections outside the staged `eigh_tred2_reduce` hunk. `rch` refused remote fmt as a non-compilation command under `RCH_REQUIRE_REMOTE=1`.

Residual and next primitive:

- The shared values path is still `7.8526 ms` median on `ts1`. The next deeper primitive is a real safe-Rust LAPACK-class eigensolver step: a blocked tridiagonalization/D&C design with an explicit floating-point parity contract, eigenvalue ordering/tie ledger, eigenvector sign/orientation proof, and fallback to the current exact EISPACK path when the proof gate cannot be satisfied.
