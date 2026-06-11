# Pass 2 Rejection: Public Full-Lower Values TRED2

## Lever

Switch `eigvalsh_contiguous_f64` from packed lower storage to the existing
bit-exact full-row-major lower reducer:

- before: copy lower triangle into packed storage, run `eigh_tred2_values_only`;
- candidate: copy lower triangle into full `n x n` lower storage, run
  `eigh_tred2_values_only_full_lower`.

The source hunk was removed after the rebench.

## Baseline

Same-worker baseline from `vmi1227854`:

| Row | Baseline p50 |
| --- | ---: |
| `eigh_f64_256x256` | `11.066 ms` |
| `eigvalsh_f64_256x256` | `6.5739 ms` |

Baseline artifact:

- `artifacts/perf/frankentorch-92yny/pass1_remote_baseline_eigh_eigvalsh_256.log`
- SHA256: `fce4a3deb8873c2be3c9000a25061b45566d30a9e3bdac1408b7994c1b766593`

## Proof

Focused RCH tests on `vmi1227854`:

- `eigh_tred2_values_only_full_lower_matches_packed_bit_exact`: passed
  - artifact: `pass2_full_lower_bit_exact_test_remote.log`
  - SHA256: `9674797e4544a6c548d432bc53d6cb67778cacd9fc7161651f25c7f1c54840ba`
- `eigvalsh_matches_eigh`: passed
  - artifact: `pass2_full_lower_eigvalsh_matches_eigh_remote.log`
  - SHA256: `54a546685f75b7e02ff41b6837b92027bf1809a8b7102addfe52f554c3bd6802`

An earlier combined filter accidentally ran zero tests because Rust test filters
do not interpret `|` as regex alternation:

- artifact: `pass2_full_lower_tests_remote.log`
- SHA256: `60079c0f05385d9ed942c0ca40c76fdbba6068e0880d0112badf91c1107aa44d`

Ordering, ties, and RNG are unchanged by construction: the final sort remains
`f64::total_cmp`, there is no RNG, and the reducer proof compares `d`, `e`, and
lower rows bit-for-bit against the packed reducer.

## Rebench

Candidate rebench on `vmi1227854`:

| Row | Candidate p50 | Result |
| --- | ---: | ---: |
| `eigh_f64_256x256` | `10.646 ms` | `1.039x` |
| `eigvalsh_f64_256x256` | `7.1546 ms` | `0.918x` |

Rebench artifact:

- `artifacts/perf/frankentorch-x53r3/pass2_full_lower_rebench_eigh_eigvalsh_256.log`
- SHA256: `27cb9972b83d7439667bd436d2a3dc543a5c3206a9e7f41a8d0afbd77b0565c5`

## Decision

Reject. The lever regresses the target values-only row and does not clear the
Score>=2.0 keep gate.

Next primitive: stop moving storage around the scalar packed sweep. Implement a
true band-packed compact-WY/dsytrd panel primitive that reduces `O(n^2 * b)`
traffic without the rejected full `n x n` WY footprint.
