# frankentorch-fql10 pass 4: parent closeout and reroute

Date: 2026-06-15
Agent: IvoryDeer

## Status

`frankentorch-fql10` has served as the umbrella bead for the general
non-symmetric `eig` / `eigvals` rewrite. The live profile-backed target remains
the serial Francis QR floor:

```text
vmi1227854 baseline after current main:
eig_f64_256x256         [53.090 ms 56.346 ms 58.649 ms]
eigvals_f64_256x256     [22.521 ms 23.155 ms 23.920 ms]
```

Pass 3 tried the only remaining non-sweep-count fql10 surface that had not
already been explicitly rejected: Hessenberg panel scratch reuse. It preserved
the strict golden digests while the hunk was present, but failed the performance
gate:

```text
unconditional scratch reuse:
eig_f64_256x256         [47.771 ms 49.804 ms 52.057 ms]
eigvals_f64_256x256     [24.173 ms 24.437 ms 24.934 ms]

full-eig-only scratch reuse:
eig_f64_256x256         [50.143 ms 53.120 ms 56.589 ms]
eigvals_f64_256x256     [23.405 ms 24.232 ms 25.134 ms]
```

Both variants failed because the shared values-only QR floor regressed, and the
full `eig` row did not retain a decisive isolated win after narrowing.

## Rejected families now covered

- threshold-only AED and whole-window AED gates;
- AED-derived alternate shift lists under the strict digest contract;
- hidden AED/window-record diagnostics linked into the public path;
- row/index/branch/range micro-cuts of the current scalar loop;
- far-row same-sweep scheduling/tape variants;
- same-order tiled sweep dispatch;
- Hessenberg allocation/scratch-shape changes.

## Closeout decision

Close this umbrella bead as rejected/rerouted. The next non-symmetric eigensolver
work must be a narrower true sweep-count primitive, not another fql10 parent
sub-pass:

- copied active-window proof harness for small-bulge multishift QR; then
- explicit shift-list handoff plus BLAS-3 far updates once the harness proves
  the strict fallback contract.

Strict fallback contract remains unchanged:

```text
n=64  eigvals=0xbc0583d464b1a211 eig=0xbc0583d464b1a211
n=128 eigvals=0x763c4b15d92c4b89 eig=0x763c4b15d92c4b89
n=256 eigvals=0x00b87b4996340204 eig=0x00b87b4996340204
stdout sha256=24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725
```

No production source diff is retained for this pass.
