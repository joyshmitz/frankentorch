# Order-statistic quickselect: flat family DONE; mode rejected; dim variant is next

Date 2026-06-13  Agent BlackThrush

## Shipped (genuine O(n log n) -> O(n) wins, "different complexity class")
- kthvalue (kgs4.57): full index-sort -> select_nth_unstable_by + O(n) exact-index
  resolution (incl ties). 9.26x same-process A/B (n=1e6).
- median + flat quantile (kgs4.58): compose on kthvalue. median 9.26x, quantile(0.5)
  3.82x. All median/quantile/nanquantile tests pass.

## REJECTED this session: mode via hash-count
tensor_mode sorted each slice O(d log d) only to group equal values for counting.
A HashMap count is O(d), and I reproduced the exact tie-breaks bit-for-bit (smallest
value among max-count wins; index = largest original index in the run; +-0.0 share a
key; NaN slices fall back to sort). All edge-case isomorphism tests passed. BUT the
bare-slice A/B (n=1e6) only hit 1.09x (card 10) / 1.68x (card 1000) / 1.33x (card 1e5):
Rust's std sort (driftsort/pdqsort) is very fast on the LOW-cardinality data that mode
realistically sees, and std HashMap's SipHash per-element overhead eats the asymptotic
edge. A per-slice HashMap alloc also risks REGRESSING the common many-small-slices
shape. Sub-2.0 + regression risk -> not shipped (reverted).
  -> To beat the sort by >=2x mode would need a faster hasher (rustc_hash/ahash — NOT a
     current dep) or a radix/bucket count (only valid for bounded integer-like values).
     mode is also niche, so low priority.

## NEXT TARGET (named with ratio): per-slice quickselect kernel for the DIM variants
tensor_quantile_dim (and a missing tensor_kthvalue_dim) sort the WHOLE tensor along
`dim` (per-slice O(d log d)) just to pick 1-2 order statistics. The kernel
`topk_tensor_contiguous_f64` ALREADY does per-lane select_nth_unstable in parallel and
documents that its (value, index) output is bit-for-bit equal to a stable sort. Mirror
it as `kthvalue_dim_tensor_contiguous_{f64,f32}` (1 value+index per lane via select_nth
at rank k-1, parallel over lanes, strided via the (outer,dim,inner) layout). Then wire
quantile_dim like the flat path + like tensor_mode: kernel gives the per-slice order-
statistic INDICES (non-diff) -> tensor_gather(input, dim, index) for autograd-aware
values -> lerp per interpolation mode. Target ~3-4x for large `dim` (matches the flat
3.8x), bit-exact (total_cmp == tensor_sort order, already proven by the flat quantile
tests). Kernel lives in the order-statistic region of ft-kernel-cpu (topk/sort), not
the GEMM peer's area.
