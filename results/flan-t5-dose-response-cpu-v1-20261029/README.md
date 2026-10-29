# FLAN-T5 public dose-response follow-up: CPU result

The prespecified aggregate was **inconclusive**. All six fixed cells were baseline-unsupported, so the 36 forwards were clean or context-only: no intervention, dose-response curve, or matched-position comparison was run. This result therefore does not test, confirm, or reject the proposed intervention hypothesis.

## Fixed question and eligibility

The [protocol](https://github.com/EastonSell/model-looking-glass/blob/ecdbf5621623878de63db432fce413de2ddb7c0a/docs/FLAN_T5_DOSE_RESPONSE_V1.md) and [case manifest](https://github.com/EastonSell/model-looking-glass/blob/ecdbf5621623878de63db432fce413de2ddb7c0a/docs/FLAN_T5_DOSE_RESPONSE_CASES_V1.json) were independently observed at public commit `ecdbf5621623878de63db432fce413de2ddb7c0a` before the operator's recorded run. For each source, the margin is the full teacher-forced log score of the first fixed German target minus that of the second, including EOS. A cell qualified for intervention only when the first-source margin was positive, the second-source margin was negative, and their difference was at least one nat. Otherwise the six clean/context forwards were retained and the cell was skipped. The recorded public observation and operator observation are not trusted clocks or independent proof of execution.

Had a cell qualified, the study would have compared word, instruction and punctuation positions at common absolute L2 doses of one-third, two-thirds and all of the minimum source-pair distance, using the fixed rules, controls and outcome thresholds in the protocol. These comparisons remain unobserved here.

## Results

| Pair (first / second) | Template | First-source margin | Second-source margin | Context-only margin | Status |
|---|---|---:|---:|---:|---|
| sun / moon | translate | 9.81515 | 5.05958 | 2.24445 | baseline unsupported |
| sun / moon | question | 7.48234 | 5.03844 | 5.13933 | baseline unsupported |
| milk / bread | translate | 13.07725 | 5.33506 | 4.80926 | baseline unsupported |
| milk / bread | question | 11.91500 | 6.88732 | 7.42742 | baseline unsupported |
| tree / flower | translate | 6.28017 | 3.24421 | 3.50691 | baseline unsupported |
| tree / flower | question | 5.78017 | 3.67805 | 3.46232 | baseline unsupported |

Every second-source margin was positive, so every pair was unevaluable and the aggregate was inconclusive. This compares two specified sequence scores; it is not a claim about greedy translations. The question context-only prompt omits the question mark along with the word, so it is a descriptive reference rather than an otherwise matched sentence. In every pair the first target has two token IDs including EOS and the second has three. The sequence-length asymmetry and target priors can affect these margins, but this experiment cannot establish why its baselines were unsupported. The examples and templates are public, correlated and exploratory; there is no population or generalization claim.

## Evidence and checks

The closed [run](run/) retains 36 full raw forward payloads, their records, case summaries and the public source observation. The terminal record `run/completed.json` has SHA-256 `253a70bd1e722f6ec642961c1385681386c2be28caff13416001a3dd1a54dbde`. The separate [operator observation](operator-observation.json) reports exit zero and binds the started and terminal records. The saved environment reports CPU float32, one thread, deterministic eager execution and `MKL_CBWR=COMPATIBLE`; the runner required the offline flags. Observed peak working set was 1,036,189,696 bytes.

The [full saved-byte replay](audit/full-saved-replay.json) checked the source binding, 36 raw payloads, recorded float32 scores, eligibility and aggregate. Its portable [auditor source](audit/full-saved-replay.py) was published at [commit 83b799e](https://github.com/EastonSell/model-looking-glass/commit/83b799e0ad912351d0c6897749d9da226cc81403) after execution and before its author inspected the real output. An independently implemented [baseline and score audit](audit/independent-baseline.json), with its [original local probe](audit/independent-baseline.py), reconstructed the EOS-inclusive scores, source/target mappings, unsupported decisions and absence of patch records; a separate [same-runtime float32 probe](audit/torch-score-replay.py) reproduced all 90 selected token scores bit-for-bit. The latter two original probes use paths relative to the reviewer's workspace; use the full auditor for a portable replay. The maximum binary64-versus-saved-float32 selected-token discrepancy was approximately `2.05e-5` nat. These are checks of retained evidence, not independently observed model execution.

The [publication index](publication-index.json) lists the size and SHA-256 of each included file other than the index itself. Model weights are not redistributed.
