# frankentorch-kgs4 cod-a fair-alloc gauntlet summary

Lever: default-off `ft-api` feature `fair-alloc`, which installs `mimalloc`
as the global allocator only for `pytorch_gauntlet_bench` when explicitly
requested with `--features fair-alloc`.

Workload: `gauntlet_avg_pool1d_grad` f64 `[8,64,8192]`, ordinary and fused
scalar-sum FT rows versus a local PyTorch CPU sidecar.

Baseline system allocator:
- `vmi1152480`: ordinary `73.690 ms`, fused `54.903 ms`.
- `vmi1152480` retake: ordinary `88.030 ms`, fused `59.073 ms`.

PyTorch sidecar:
- Five 40-iteration totals: `0.524018352968`, `0.527352298028`,
  `0.465811557951`, `0.505421738024`, `0.540594351944` seconds.
- Median: `13.100458824 ms/iter`.

Fair allocator:
- `ovh-a`: ordinary `11.830 ms`, fused `13.417 ms`.
- `hz2`: ordinary `22.545 ms`, fused `8.4890 ms`.

Win/loss/neutral:
- Default FT: `0W / 2L / 0N`.
- Fair-alloc FT: `2W / 1L / 1N` across two workers.

Caveat: no same-worker system-vs-fair A/B landed; RCH selected `vmi1152480`
for default runs, `ovh-a` and `hz2` for fair runs. Keep is for a default-off
fair comparison feature, not a default product-speed claim.

Gates:
- `rustfmt --edition 2024 --check crates/ft-api/benches/pytorch_gauntlet_bench.rs`: passed.
- `git diff --check`: passed.
- `cargo metadata --locked --manifest-path crates/ft-api/Cargo.toml --features fair-alloc`: passed.
- `rch exec -- cargo check -p ft-api --features fair-alloc --bench pytorch_gauntlet_bench`: passed.
- `rch exec -- cargo clippy -p ft-api --features fair-alloc --bench pytorch_gauntlet_bench -- -D warnings`: passed.
- `rch exec -- cargo test -p ft-conformance --profile release`: passed.
