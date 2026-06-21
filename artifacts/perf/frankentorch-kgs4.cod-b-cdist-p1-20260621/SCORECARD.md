# cdist p=1 f64 head-to-head

Date: 2026-06-21
Assignee: cod-b
Agent: IvoryDeer
Target dir: `/data/projects/.rch-targets/frankentorch-cod-b`

## Verdict

No-ship. FrankenTorch's fused p=1 Manhattan `cdist` kernel is correct on the tested fixture, but PyTorch is
faster on the same host. A small inner-loop slice/zip indexing lever was tested and rejected; no product
source is kept from this probe.

The radical lever considered here came from cache/indexing pressure: keep the fused no-materialization
kernel, but remove repeated base-plus-index address arithmetic and bounds checks from the p=1 inner loop.
The trial did not improve the measured path.

## Same-host ratio source

Command source: retrieved warm-target FrankenTorch release example binary plus local PyTorch CPU.

Log: `local_binary_cdist_p1_headtohead.log`

| Lane | FrankenTorch | PyTorch | Ratio | Correctness |
| --- | ---: | ---: | ---: | --- |
| `cdist(x1, x2, p=1.0)` f64 `[1024,128] x [1024,128]` | 15.292 ms | 8.782 ms | FT 1.74x slower | checksum matches rounded FT print |

PyTorch checksum check: `5.440050439680e+07`. FrankenTorch printed checksum: `5.4401e7`.

Scorecard: 0W / 1L / 0N.

## RCH evidence

Baseline RCH FT-only run: `rch_cdist_p1_headtohead.log`

- Worker: `ovh-a`
- FrankenTorch: 19.288 ms
- PyTorch: unavailable on worker
- Note: RCH rewrote `CARGO_TARGET_DIR` to a worker-scoped path and paid a cold compile, so this is routing
  evidence only.

Rejected slice/zip candidate RCH FT-only run: `rch_cdist_p1_slicezip_candidate.log`

- Worker: `hz2`
- FrankenTorch: 20.138 ms
- PyTorch: unavailable on worker
- Candidate local binary could not run on this host because it required `GLIBC_2.43`.
- Verdict: rejected from remote routing evidence; source hunk reverted.

## Behavior Preservation

- Invariant behavior preserved by kept tree: yes; no product source changed.
- Trial hunk accumulation order: left-to-right, intended to preserve numeric behavior.
- Trial hunk status: reverted before commit because it did not improve the measured path.

## Conformance

Gate: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b AGENT_NAME=IvoryDeer rch exec -- cargo test -p ft-conformance --profile release`

Log: `test_ft_conformance_release.log`

Result: green on `vmi1153651`; 199 `ft_conformance` lib tests plus conformance bins, integration tests,
smoke tests, and doctests passed.

## Retry Predicate

Do not retry p=1 `cdist` with indexing-only or iterator-shape micro-levers. A credible retry needs a deeper
SIMD/tiled Manhattan kernel that can beat PyTorch's vectorized CPU path, with exact checksum reporting in the
head-to-head harness before scoring.
