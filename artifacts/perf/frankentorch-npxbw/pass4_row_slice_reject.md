# Pass 4 - Row-Sliced Small-Bulge Kernel Rejected

Bead: `frankentorch-npxbw`

## One Lever Tried

Tried a bounded small-bulge access-pattern microkernel inside the existing Francis double-shift chase:

- row modification used row slices instead of repeated `h[row * n + col]` indexing
- column modification used row slices over the same row range
- arithmetic order for each updated slot was preserved
- no public dispatch or eigenvalue ordering policy changed

The source hunk was removed after the same-worker performance gate failed.

## Baseline

Criterion via RCH on actual worker `vmi1153651`:

- `eigvals_f64_256x256`: `[56.435 ms 58.071 ms 60.030 ms]`
- `eig_f64_256x256`: `[92.077 ms 95.486 ms 99.420 ms]`

The first baseline log file is named with the requested worker, but RCH selected `vmi1153651`; the log footer is authoritative.

## Candidate

Criterion via RCH on `vmi1153651`:

- `eigvals_f64_256x256`: `[65.383 ms 70.937 ms 77.362 ms]`

Median ratio: `58.071 / 70.937 = 0.819x`. Since the shared eigvals floor regressed, `eig_f64_256x256` candidate rebench was not run.

## Behavior Proof

While the candidate hunk was applied:

- `rch exec -- cargo test -p ft-kernel-cpu --lib eig` passed 21 focused eig/eigh tests on `vmi1153651`.
- `rch exec -- cargo run -q -p ft-kernel-cpu --example eigvals_golden` passed on `vmi1153651`.
- Strict golden SHA-256 matched pass 3 exactly: `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.

## Verdict

Rejected. Score `0.0`.

This pass confirms that storage-access reshaping of the current single-bulge loop is not the next route. Pass 5 should attack an algorithmically different primitive: deterministic small-bulge multishift shift-list handoff or a BLAS-3 far-update tile, not another row/indexing microkernel.
