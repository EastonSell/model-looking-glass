# Screened projected-edit follow-on: results

The supervised discovery-only stage is complete. It used the earlier public discovery prompt and four calibration prompts; none of the 32 new target-screen prompts or 12 locality prompts was forwarded in that stage. The subsequent baseline screen and edited evaluation are also complete. The discovery measurements are not results on the new cases.

The fixed discovery grid selected the unprojected dose **0.00075**, or 0.075% of the original edited-weight norm. The projected dose is 0.001, or 0.1%. The original discovery London-minus-Paris margin was −1.268082; the projected edit increased it by 1.175011 and the matched unprojected edit increased it by 1.165703. Their gain difference, 0.009308, is below the prespecified tolerance 0.117501. Both edited margins remain negative. The realized matched unprojected update norm was 0.230065 versus 0.306753 for projected. The match is on this one previously exposed discovery prompt and does not establish matched efficacy on the new targets.

| Unprojected planned norm fraction | Discovery margin gain |
| --- | ---: |
| 0.000500 | 0.777115 |
| 0.000625 | 0.971413 |
| **0.000750** | **1.165703** |
| 0.000875 | 1.359993 |
| 0.001000 | 1.554276 |
| 0.001125 | 1.748520 |
| 0.001250 | 1.942780 |

The projection uses all 18 calibration-token activations; its retained rank was 17. The maximum observed calibration-site output shift was approximately 0.0000229 for projected and 0.252833 for the discovery-matched unprojected dose. These same activations define the projection, so this is a construction check, not an independent locality outcome. The equal-norm unprojected, projected-random, opposite, and nuisance controls and all seven dose candidates are retained in the evidence. The projected-random and nuisance controls are single fixed draws.

The [candidate](development/candidate/pre-evaluation-candidate.json) and [phase record](development/phase-record.json) inventory all raw logits, calibration outputs, algebraic effects, factors, gradients, states and attestations. An independent reviewer reconstructed the saved math from the pinned checkpoint bytes and artifacts; the [math audit](audits/development-math.json) and [saved custody audit](audits/development-custody.json) describe their scope. The source and runtime are bound by the [published manifest](../../experiments/targeted_edit_screened_run_v1.json); the [protocol](../../docs/TARGETED_EDIT_SCREENED_MATCHED_V1.md) and [case-scope note](../../docs/TARGETED_EDIT_SCREENED_CASES.md) define the remaining gates and limitations. The recorded child and supervisor identities agree, and the reports state clean exit, reaping, job drain, restoration and model disposal. These are cooperative execution and cleanup attestations, not authenticated proof against concurrent mutation.

The saved v18 public-inventory attestation records a local timestamp of 07:57:32.6912981 UTC. Its named reviewer did not independently attest that exact fractional instant; they report that the check completed no later than their 07:58:06 UTC manifest-review observation, before the recorded development start. The original receipt is retained unchanged. Those reported clock values and Git ancestry do not authenticate real-world chronology. The public comparison and custodian checks cover only known histories and the named protected split; unknown exposure and pretraining remain unknown.

## Baseline screen

The original model was run on all 32 fixed public target prompts. **Twenty-five had a strictly negative London-minus-Paris margin**, and the frozen round-robin rule selected eight: offsets 0, 8, 16, 24, 1, 11, 17 and 26, with two from each stratum. This satisfies the target-count eligibility condition; it is not an edited-evaluation outcome. All 32 margins, eligibility flags, selected IDs and per-row hashes are in the [selection record](screen/screen/screen-selection.json), with the [full-vocabulary baseline rows](screen/screen/screen.f32), [ordered observations](screen/screen/observations.jsonl) and [phase record](screen/phase-record.json). The selected targets are public and baseline-conditioned, not blind or representative of all paraphrases.

On all 32 baselines, neither London nor Paris was the full-vocabulary top token. A reversal of the London-versus-Paris margin therefore shows a change in their pairwise preference; it does not by itself show that the model's top prediction or a generated answer became London. The subsequent edited evaluation is reported below.

Two independent model-free reviews checked the saved screen: the [selection/math report](audits/screen-math.json), [process-record review](audits/screen-custody.json), and [scalar per-row audit](audits/screen-scalar.json) (with its [post-run source](audits/screen-scalar.py)). These verify saved bytes and reported consistency, not authenticated execution or cleanup. The screen's [manifest](../../experiments/targeted_edit_screened_screen_run_v1.json) binds the published discovery record and case pool. The saved v20 inventory and named custodian reports were checked before the recorded start; v21 added only separately checked planned inputs. GitHub reports that v21 merged at 08:20:56 UTC, after the screen's recorded 08:20:36 UTC start, but these clocks are not independently authenticated or synchronized. The inventory and private-split checks do not establish disjointness from unknown pretraining or exposure.

## Edited evaluation

**Both prespecified comparisons are negative.** The projected edit reversed one of eight selected London-minus-Paris margins (the rule requires at least two), and one of 12 locality top token IDs changed (the rule requires all 12 unchanged). The other projected target, locality and KL criteria passed; those two failures determine both decisions. These are exploratory, baseline-conditioned public cases, not a blind or representative sample.

| Condition | Mean target gain | Reversals / 8 | Mean locality KL | Max locality KL | Mean expected-NLL change | Max expected-NLL increase | All locality tops unchanged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Original | 0.000000 | 0 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000 | yes |
| Projected (0.1%) | 0.717697 | 1 | 0.000276719 | 0.000892819 | 0.002699 | 0.192730 | no |
| Unprojected, equal norm (0.1%) | 1.074435 | 2 | 0.000767718 | 0.002154482 | 0.007413 | 0.301922 | no |
| Unprojected, discovery-matched (0.075%) | 0.805818 | 1 | 0.000429586 | 0.001181551 | 0.005407 | 0.226252 | no |
| Projected random | 0.000026 | 0 | 0.000000339 | 0.000001349 | 0.000145 | 0.000972 | yes |
| Opposite projected | -0.717485 | 0 | 0.000269665 | 0.000798025 | -0.002087 | 0.406799 | yes |
| Nuisance-span control | 1.070867 | 2 | 0.000763886 | 0.002145358 | 0.007237 | 0.299523 | no |

The projected mean target gain was 0.717697, versus 1.074435 for the equal-norm unprojected edit and 0.805818 for the discovery-matched edit. The latter's evaluated-target mean gap was 0.088121, within the preset 0.201454 tolerance, but projected gain was about 10.9% smaller. Matching was performed on the old discovery prompt, and this broad tolerance does not establish equal efficacy on the selected targets. The lower projected locality KL is a descriptive comparison, not proof of superior preservation at equivalent strength. The fixed random and nuisance controls are single draws.

The changed projected locality top was ` euro` (token 11063) to ` Euro` (token 1898), a capitalization change in the same currency name. It fails the frozen exact-token rule but does not by itself show loss of currency knowledge. The equal-norm unprojected edit also changed the France-continent row's top from ` France` to ` the`; that row's expected continuation was ` Europe` and was already not top at baseline. Only 3 of 12 locality expected continuations were baseline top tokens. The projected mean expected-NLL change of +0.002699 masks individual changes, including +0.192730 on the Paris locality and −0.404623 on the UK-capital locality; all rows are retained.

Neither London nor Paris was the full-vocabulary top token on any of the eight selected targets under any condition. A reversed pairwise margin therefore does not show a London top prediction or generated answer. The eight target gains are:

| Target ID | Original margin | Projected gain | Equal-norm gain | Discovery-matched gain | Projected margin | Original → projected top ID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| screen-target-s1-01 | -3.263580 | 0.711906 | 1.068893 | 0.801590 | -2.551674 | 262 → 262 |
| screen-target-s2-01 | -0.133198 | 0.834610 | 1.195736 | 0.896866 | 0.701412 | 262 → 262 |
| screen-target-s3-01 | -1.099525 | 1.011024 | 1.374321 | 1.030785 | -0.088501 | 783 → 783 |
| screen-target-s4-01 | -2.684467 | 0.200661 | 0.488861 | 0.366631 | -2.483807 | 326 → 326 |
| screen-target-s1-02 | -3.313957 | 0.758472 | 1.115555 | 0.836601 | -2.555485 | 262 → 262 |
| screen-target-s2-04 | -1.629562 | 1.238190 | 1.594147 | 1.195625 | -0.391373 | 1363 → 1363 |
| screen-target-s3-02 | -2.326584 | 0.964607 | 1.354912 | 1.016151 | -1.361977 | 262 → 262 |
| screen-target-s4-03 | -0.600914 | 0.022102 | 0.403053 | 0.302292 | -0.578812 | 25 → 25 |

The [target CSV](target-metrics.csv) contains all 56 target/condition rows, including both candidate NLLs, KL and top IDs. The [locality CSV](locality-metrics.csv) contains all 84 locality/condition rows, including expected-token NLL, both candidate NLLs, margin, KL and top IDs. Full float32 logits, the ordered row journal, state checks, provisional criteria and process reports are inventoried by the [evaluation phase record](evaluation/phase-record.json).

The [independent scientific reader](audits/evaluation-scientific.py) and [report](audits/evaluation-scientific.json) recompute all seven conditions and both criteria from saved rows; a [separate saved-evidence and process audit](audits/evaluation-independent.json) checks the phase inventory, baseline reuse, fixed updates and reported cleanup. These checks establish consistency of retained bytes and reports, not authenticated execution, chronology, cleanup or disjointness from unknown pretraining. The model owner reported clean exit, reaping, job drain, restoration and disposal.

From the repository root, the scientific reader can be replayed with Python and NumPy (the report path must not already exist):

```text
python results/targeted-edit-screened-matched-v1-20261029/audits/evaluation-scientific.py \
  --cases experiments/targeted_edit_screened_public_cases_v1.json \
  --screen-dir results/targeted-edit-screened-matched-v1-20261029/screen/screen \
  --candidate results/targeted-edit-screened-matched-v1-20261029/development/candidate/pre-evaluation-candidate.json \
  --data-dir results/targeted-edit-screened-matched-v1-20261029/evaluation/evaluation \
  --report new-audit.json
```

The frozen owner's internal exposure field records v23. Before evaluation, an [independent external v24 comparison](../../experiments/targeted_edit_screened_evaluation_v24_comparison_v1.json) verified the same 547 known input rows and 3,888 prefixes, all 44 exact study records and no known bidirectional prefix collision; its [pre-forward receipt](../../experiments/targeted_edit_screened_evaluation_v24_check_v1.json) explains this distinction. An earlier v24-merged checkout was rejected by the source-hash check before an attempt or model load; the unrelated source update was not reverted. Custodian report times in the receipt are operator-recorded no-later-than bounds. These public and named-custodian checks cannot certify unknown history.
