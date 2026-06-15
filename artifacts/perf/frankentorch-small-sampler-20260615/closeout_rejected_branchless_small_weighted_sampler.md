# frankentorch-zcrzg closeout: rejected branchless small weighted sampler

Status: rejected/no code kept.

Baseline, RCH `ovh-a` (`pass1_baseline_sampler_vmi1227854.log`; worker selected `ovh-a`):
- `sampler/without_replacement_repeated_passes_4096x256`: `[4.5863 ms 4.6180 ms 4.6392 ms]`
- `sampler/weighted_two_positive_2x1m`: `[3.6649 ms 3.7040 ms 3.7253 ms]`
- `sampler/weighted_three_positive_3x1m`: `[3.8140 ms 3.8393 ms 3.8601 ms]`
- `sampler/weighted_four_positive_4x1m`: `[4.4099 ms 5.2403 ms 5.9297 ms]`
- `sampler/weighted_4096_positive_4096x262k`: `[2.2565 ms 2.3578 ms 2.4235 ms]`

Candidate:
- One lever tried: replace len==2 and len==3 weighted-sampler branch chains with strict-threshold-count helpers analogous to the existing len==4 helper.
- Isomorphism argument: strict `u > threshold` counts preserve `u <= threshold` lower-bucket ties, collapsed zero-weight thresholds, RNG order, output order, validation order, and error behavior.

Proof:
- Focused test: `rch exec -- cargo test -j 1 -p ft-data weighted_sampler -- --nocapture`
- Result: 7 weighted sampler tests passed, including existing golden fixture tests and collapsed-threshold edge coverage.
- Golden SHA:
  - `ft_data_weighted_sampler_branchless_frankentorch-sx6l.txt`: `5139c7e647c38c1f9000e022197f10be6426544e9561ea724e0c60fa08062d90`
  - `ft_data_weighted_large_cardinality_frankentorch-j54u.txt`: `ff33133ed2d4dab1627878e7ba7f7d1fe4c426a064be2749cd75a740b210b3e8`

After, RCH `ovh-a` (`pass3_rebench_sampler_branchless_ovh_a.log`):
- `sampler/without_replacement_repeated_passes_4096x256`: `[4.9903 ms 5.1705 ms 5.4177 ms]`
- `sampler/weighted_two_positive_2x1m`: `[4.1757 ms 4.5252 ms 4.9179 ms]`
- `sampler/weighted_three_positive_3x1m`: `[3.8866 ms 4.0745 ms 4.2326 ms]`
- `sampler/weighted_four_positive_4x1m`: `[4.3442 ms 4.4114 ms 4.4933 ms]`
- `sampler/weighted_4096_positive_4096x262k`: `[2.2149 ms 2.2570 ms 2.2775 ms]`

Decision:
- Rejected. The affected two- and three-weight rows regressed (`3.7040 -> 4.5252 ms`, `3.8393 -> 4.0745 ms` medians).
- Score `< 2.0`; code restored to pre-candidate state before closeout.
