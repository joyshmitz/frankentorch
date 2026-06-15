# frankentorch-2lmvy IvoryDeer revalidation

Source commit already present at session closeout:

- `f9e6fb4c59f42b6c4cc109916cba47cd0a5bd273`
- `perf(ft-api): parallelize ndtr/log_ndtr/ndtri family forward+backward + skip dead no-grad saves`

Focused rch harness:

- Command: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-a CARGO_TERM_COLOR=never rch exec -- cargo run -q --release -p ft-api --example ndtr_family_ab`
- Worker selected by rch: `ovh-a`
- Harness: `crates/ft-api/examples/ndtr_family_ab.rs`
- Input size: `n=4_000_000`

Measured A/B from `ndtr_family_ab_ovh_a.log`:

- `ndtr`: old serial map plus dead save clone `50.04 ms`; new production tensor path `21.37 ms`; `2.34x`
- `log_ndtr`: old serial map plus dead save clone `74.24 ms`; new production tensor path `25.65 ms`; `2.89x`

Behavior proof:

- Harness asserts `ndtr` and `log_ndtr` outputs bit-for-bit against serial scalar references before timing.
- Harness asserts `ndtri` output bit-for-bit against a serial chunk-concatenated reference, where chunks remain below `PARALLEL_ELEMENTWISE_MIN`.
- Ordering is preserved because `par_map_f64` and `par_zip_map_f64` write indexed output positions.
- No RNG, tie-breaking, or reduction-order behavior is introduced.
- Golden manifest check passed: `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`.

Alien primitive:

- Applied the §8.2 vectorized/morsel-style execution idea to independent special-function tensor lanes.
- Fallback contract remains the existing serial path below `PARALLEL_ELEMENTWISE_MIN`; grad/create_graph formulas are unchanged.

Score:

- Impact 4, confidence 4, effort 2 => `8.0`; keep.
