# frankentorch-8ubk3 Pass 1 Baseline/Profile

Date: 2026-06-12

Scope: baseline/profile for successor bead `frankentorch-8ubk3`.

## Criterion Rows

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eigvals_f64_256x256
worker: ovh-a
eigvals_f64_256x256 time: [23.550 ms 23.596 ms 23.646 ms]

RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eig_f64_256x256
worker: hz2
eig_f64_256x256 time: [53.025 ms 53.634 ms 54.269 ms]
```

The first attempt in `pass1_bench_eigvals_256.log` reached benchmark startup
but did not contain a measured row; the appended rerun above is the usable
baseline. The first attempt in `pass1_bench_eig_256.log` also reached benchmark
startup without a measured row; the appended rerun above is the usable baseline.

Because the two Criterion rows landed on different RCH workers, pass 4 or later
must run an immediate same-worker before/after pair before any source hunk can
qualify as a keep.

## Golden Output

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo run --release -q -p ft-kernel-cpu --example eigvals_golden
worker: ovh-a
strict stdout sha256: 24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725
n=64  eigvals_digest=0xbc0583d464b1a211 eig_digest=0xbc0583d464b1a211
n=128 eigvals_digest=0x763c4b15d92c4b89 eig_digest=0x763c4b15d92c4b89
n=256 eigvals_digest=0x00b87b4996340204 eig_digest=0x00b87b4996340204
```

The strict SHA is over only the golden stdout lines:

```text
grep -E '^(frankentorch-l9xod eigvals_golden|eigvals_digest=|eig_digest=)' pass1_eigvals_golden.log | sha256sum
```

## Timing Probe

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo run --release -q -p ft-kernel-cpu --example eig_timing_probe
worker: ovh-a
n=128  eigvals=4.15ms    eig=5.61ms    sweeps=173  defl1=28 defl2=50  fallback=0 exceptional=0 max_width=128
n=256  eigvals=29.42ms   eig=39.49ms   sweeps=319  defl1=14 defl2=121 fallback=0 exceptional=0 max_width=256
n=512  eigvals=311.44ms  eig=408.97ms  sweeps=583  defl1=10 defl2=251 fallback=0 exceptional=0 max_width=512
n=1024 eigvals=1979.73ms eig=3285.78ms sweeps=1132 defl1=18 defl2=503 fallback=0 exceptional=0 max_width=1024
```

First shift samples:

```text
n=256  [0..255 x=1.290e2 y=1.240e2 w=6.569e1 exceptional=false]
n=1024 [0..1023 x=5.118e2 y=5.120e2 w=1.282e2 exceptional=false]
```

## Proof Contract Seed

- Ordering/tie behavior: unchanged baseline only; source edits must preserve the
  current bottom-up slot policy and complex-pair slot convention.
- Floating-point/RNG: no source edits; no RNG path is present in the probe.
- Shift stream: source edits must prove identical `EigFrancisShiftSample`
  sequence or fall back before changing output.
- Deflation/accounting: source edits must preserve selected `m`, `defl1`,
  `defl2`, fallback count, exceptional count, and max-width accounting.
- Golden output: source edits must keep strict stdout SHA
  `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.

No source files were edited.
