# frankentorch-x9137 Pass 1 Baseline/Profile

Date: 2026-06-13T00:30:00Z
Bead: `frankentorch-x9137`
Target: strict scalar-shift-preserving shadow active-window blocked Francis sweep proof harness.

## RCH / Scope

- All commands were crate-scoped to `ft-kernel-cpu`.
- All commands used `RCH_REQUIRE_REMOTE=1`.
- Worker selected for all decisive rows: `hz2`.
- No local fallback occurred.

## Criterion Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eigvals_f64_256x256 --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Result on `hz2`:

```text
eigvals_f64_256x256     time:   [26.396 ms 26.500 ms 26.737 ms]
```

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eig_f64_256x256 --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Result on `hz2`:

```text
eig_f64_256x256         time:   [53.720 ms 55.281 ms 57.322 ms]
```

## Golden Output

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo run -q -p ft-kernel-cpu --release --example eigvals_golden
```

Worker: `hz2`

Strict stdout SHA-256:

```text
24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725
```

Printed digest anchors:

```text
n=64   eigvals=0xbc0583d464b1a211 eig=0xbc0583d464b1a211
n=128  eigvals=0x763c4b15d92c4b89 eig=0x763c4b15d92c4b89
n=256  eigvals=0x00b87b4996340204 eig=0x00b87b4996340204
```

## Francis Profile

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo run -q -p ft-kernel-cpu --release --example eig_timing_probe
```

Worker: `hz2`

```text
threads=16
n=128   eigvals=7.35ms    eig=11.57ms    vec_machinery=4.21ms
profile n=128   sweeps=173  defl1=28  defl2=50   fallback=0  exceptional=0  max_width=128   samples=173   truncated=false
n=256   eigvals=37.37ms   eig=58.23ms    vec_machinery=20.86ms
profile n=256   sweeps=319  defl1=14  defl2=121  fallback=0  exceptional=0  max_width=256   samples=319   truncated=false
n=512   eigvals=432.96ms  eig=530.50ms   vec_machinery=97.54ms
profile n=512   sweeps=583  defl1=10  defl2=251  fallback=0  exceptional=0  max_width=512   samples=583   truncated=false
n=1024  eigvals=2840.52ms eig=4841.54ms  vec_machinery=2001.02ms
profile n=1024  sweeps=1132 defl1=18  defl2=503  fallback=0  exceptional=0  max_width=1024  samples=1132  truncated=false
```

## Diagnosis

The profile confirms the `frankentorch-x9137` target remains the non-symmetric Francis QR floor. The `eigvals` path has no eigenvector machinery and still spends 319 scalar double-shift sweeps at n=256 and 1132 at n=1024 on the benchmark fixture. There are no fallback or exceptional shifts on the target rows, so a strict scalar-shift-preserving shadow harness can compare the active-window, shift, selected-m, and deflation streams without introducing alternate shift policy.

## Pass 2 Gate

Proceed to the alien primitive/proof-contract pass.

- Allowed next primitive: private shadow active-window blocked/tiled row-column ledger harness.
- Not allowed: range/index micro-cuts, alternate shift packets, diagnostic-only helpers, or public `eig_impl` dispatch changes.
- Baseline comparator for any later keep: same-worker `hz2` rows above, or a fresh immediate same-worker before/after pair if RCH selects another worker.
