# frankentorch-v6kaj - rejected Mish borrowed-input lever

## Target

- Fresh profile target: `activation_backward/mish_chain_16x65536`.
- Baseline command:
  `RCH_REQUIRE_REMOTE=1 RCH_WORKERS=vmi1227854 rch exec -- cargo bench -j 1 -p ft-autograd --bench backward_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20`
- Baseline artifact:
  `artifacts/perf/frankentorch-v6kaj/pass1_baseline_backward_bench_vmi1227854.log`
- Baseline timing on `vmi1227854`: `[19.370 ms 22.463 ms 26.740 ms]`.

## One Lever

- Candidate: in `TensorNodeOp::Mish` backward, use `Self::operand_values_cow(...)` instead of `contiguous_values_as_f64()`.
- Intended effect: borrow contiguous F64 input values and avoid cloning one `Vec<f64>` per Mish node.
- Isomorphism argument: same Mish derivative formula, same ascending index order, no tie-breaking/RNG/order changes, and non-F64/non-contiguous paths retain the same converted/error behavior through `operand_values_cow`.

## Behavior Proof

- Remote proof command:
  `RCH_REQUIRE_REMOTE=1 RCH_WORKERS=vmi1227854 rch exec -- cargo test -j 1 -p ft-autograd tensor_mish -- --nocapture`
- Result: 2/2 Mish bit-exact tests passed.
- Golden proof log SHA-256:
  `96980055a9f699898eab4f2fe3279e567f27d6cbbea530b7aa0db89c6316174a`

## Rebench

- Rebench command:
  `RCH_REQUIRE_REMOTE=1 RCH_WORKERS=vmi1227854 rch exec -- cargo bench -j 1 -p ft-autograd --bench backward_bench mish_chain_16x65536 -- --warm-up-time 1 --measurement-time 5 --sample-size 20`
- Candidate timing on `vmi1227854`: `[28.508 ms 32.248 ms 36.518 ms]`.
- Delta: mean regressed `22.463 ms -> 32.248 ms` (`0.70x` baseline-normalized throughput).

## Verdict

- Score: `0.0` (`Impact=0`, `Confidence=4`, `Effort=1`).
- Source hunk was removed; do not repeat the Mish borrowed-contiguous-input family without new evidence.
- Next route: Mish needs an algorithmically different primitive, likely a bit-exact derivative-side cached forward auxiliary or a separate special-function approximation proof. A plain input borrow does not help this workload.
