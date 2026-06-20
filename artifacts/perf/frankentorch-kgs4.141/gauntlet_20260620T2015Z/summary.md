# frankentorch-kgs4.142 avg_pool1d tensor_sum shortcut

Verdict: rejected and reverted.

Note: the raw artifact directory keeps the original pre-rebase
`frankentorch-kgs4.141` path because upstream claimed `.141` for a BatchNorm
lane while this run was in progress. The final tracker/docs ID for this
avg_pool1d lane is `frankentorch-kgs4.142`.

Candidate: register f64 `functional_avg_pool1d` outputs and route ordinary
`functional_avg_pool1d(...).tensor_sum()` to the existing scalar-loss backward
when output `retain_grad` and hooks are absent.

Same-worker rch worker: `vmi1153651`.

| Row | Baseline median | Candidate median | Criterion verdict |
|---|---:|---:|---|
| ordinary `frankentorch_kgs4_122` | `1.6792 s` | `1.2016 s` | no change, `p=0.44` |
| explicit scalar-sum `frankentorch_kgs4_134_fused_sum_loss` | `810.56 ms` | `650.33 ms` | no change, `p=0.57` |

Local PyTorch comparator through `crates/ft-api/benches/pytorch_avg_pool1d_grad.py`
used five 10-iteration runs with 32 compute/inter-op threads. Median per
iteration was `11.493027 ms`.

Mixed-location ratios against that PyTorch median:

| Row | Ratio vs PyTorch |
|---|---:|
| baseline ordinary | `146.11x` slower |
| baseline explicit scalar-sum | `70.53x` slower |
| candidate ordinary | `104.55x` slower |
| candidate explicit scalar-sum | `56.58x` slower |

Focused candidate test:

```text
rch exec -- cargo test -p ft-api functional_avg_pool1d_tensor_sum --lib --profile release -- --nocapture
```

Result: `2 passed; 0 failed`.

Post-revert checks:

```text
rch exec -- cargo test -p ft-conformance strict_scheduler --profile release -- --nocapture
git diff --check
ubs docs/NEGATIVE_EVIDENCE.md docs/RELEASE_READINESS_SCORECARD.md artifacts/perf/frankentorch-kgs4.141/gauntlet_20260620T2015Z/summary.md
```

Results: focused conformance `1 passed; 0 failed`; `git diff --check` passed;
UBS exited `0` and reported no recognizable source language in the Markdown and
artifact files.

Reason for revert: the ordinary-row median improved, but Criterion reported no
statistically significant change and the explicit scalar-sum control row moved
under the same noisy worker conditions. This is not credible keep evidence.
