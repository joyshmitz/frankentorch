# FT-P2C-003 Optimization + Isomorphism Evidence (v1)

## Optimization Lever

- ID: `e2e-packet-scope-suite-gating`
- Change: packet-filtered e2e execution now runs only suites mapped to the requested packet.
- Path: `crates/ft-conformance/src/lib.rs`

## Benchmark Delta (`packet_e2e_microbench_op_schema_produces_percentiles`)

- Baseline (pre-optimization): `p50=2623117ns`, `p95=3284608ns`, `p99=3284608ns`, `mean=2774106ns`
- Post (after optimization): `p50=454150ns`, `p95=487069ns`, `p99=487069ns`, `mean=457221ns`
- Improvement: `p50=82.687%`, `p95=85.171%`, `p99=85.171%`, `mean=83.518%`

## Isomorphism Checks

- unit/property: `strict_op_schema_conformance_is_green` passed
- differential comparator presence: `differential_op_schema_adds_metamorphic_and_adversarial_checks` passed
- e2e packet filter behavior: `e2e_matrix_packet_filter_includes_op_schema_packet_entries` passed
- quality gates: workspace `cargo check`, `cargo clippy -D warnings`, and `cargo fmt --check` passed

## Memory Churn Note

- `rch` test stream does not report max RSS directly for this microbench command.
- This artifact records deterministic latency-tail improvement; memory-churn instrumentation remains deferred to dedicated perf harness integration.
