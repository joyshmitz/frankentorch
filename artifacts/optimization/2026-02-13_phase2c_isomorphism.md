# Phase-2C Isomorphism Proof Blocks (2026-02-13)

## Change: FT-P2C-002 Dispatch Key Model

- Ordering preserved: yes, deterministic priority list is explicit and test-covered.
- Tie-breaking unchanged: yes, no ambiguous tie in scoped key domain.
- Floating-point: identical (`ft-kernel-cpu` math path unchanged).
- RNG seeds: N/A.
- Golden outputs: `sha256sum -c artifacts/optimization/golden_checksums.txt` passed.
- Differential parity anchor refresh (2026-02-15): `artifacts/phase2c/conformance/differential_report_v1.json` has `failed_checks=0` with `oracle.available=true`.
- E2E replay anchor refresh (2026-02-15): `artifacts/phase2c/e2e_forensics/ft-p2c-002.jsonl` contains strict+hardened packet slice (6 entries, 0 failures).

## Change: FT-P2C-004 Autograd Scheduler

- Ordering preserved: yes, deterministic queue order is explicit and fixture-verified.
- Tie-breaking unchanged: yes, `NodeId` max-heap tie-break is deterministic.
- Floating-point: identical derivative formulas (`add`/`mul`) retained.
- RNG seeds: N/A.
- Golden outputs: `sha256sum -c artifacts/optimization/golden_checksums.txt` passed.

## Change: FT-P2C-006 Serialization + RaptorQ

- Ordering preserved: yes, checkpoint entries normalized by `node_id` before hashing.
- Tie-breaking unchanged: yes, canonical ordering removes insertion-order drift.
- Floating-point: value/grad bits are hashed and serialized directly.
- RNG seeds: deterministic fixed seed for sidecar generation (`0x4654_5f52_4150_5451`).
- Golden outputs: `sha256sum -c artifacts/optimization/golden_checksums.txt` passed.
