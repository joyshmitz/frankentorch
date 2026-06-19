# Validation

Date: 2026-06-19

Product source diff after candidate rejection: none.

Commands:

- `git diff --check`
  - Result: pass.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo check -p ft-api --benches`
  - Result: pass on rch worker `hz2`.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo test -p ft-api functional_max_pool3d_grad_matches_finite_diff --lib`
  - Result: pass on rch worker `hz1`; `1 passed; 0 failed; 2327 filtered out`.

Conformance status for the touched workload: green.
