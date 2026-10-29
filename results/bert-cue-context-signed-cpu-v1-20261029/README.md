# Signed cue-context transfer on DistilBERT

The prespecified signed rule passed in **two of three** evaluation pairs. All three pairs passed the clean-baseline and common-dose screens. The stricter sensitivity rule, requiring every recipient margin to exceed 0.05, passed in **none of three** pairs. The separately reported discovery pair failed both rules. These are fixed descriptive outcomes on a small set of public, nonblind cases; the study design was chosen after inspecting the preceding cue-context results.

The secondary analysis points in a different direction. It was evaluable in two evaluation pairs: among 12 recipient-directions with a sufficiently large same-cue clean-score gap, three shifts pointed toward the donor's clean score and none exceeded the magnitude of the matched same-class shift. The third evaluation pair had only two qualifying directions and was unevaluable under the fixed six-of-eight rule. Opposite directions share the same underlying cue gap; they are not independent examples.

## Fixed question and results

For each cue recipient, the same-cue, different-same-class and both opposite-class donors came from the **same alternate context**, at the same `cue2` site and requested L2 norm 0.5. With `L = logit(any) − logit(some)`, each effect is patched minus clean L. For a negative/low-frequency recipient (`never`, `rarely`) the class-directed sign is −1; for an affirmative/high-frequency recipient (`always`, `often`) it is +1. The primary margin is

`D = min(class-directed opposite-cue effects) − max(|same-cue effect|, |different-same-class effect|)`.

Both opposite-class donors must individually exceed both active controls: all eight recipient margins must be strictly positive for a pair to pass. Among at least two eligible evaluation pairs, the fixed aggregate requires two eligible passes. The stronger sensitivity requires all eight margins strictly greater than 0.05, with the same aggregation. A baseline is eligible only when all four negative-minus-affirmative clean L gaps exceed 0.5 in each context; the primary also requires the common dose to be exactly 0.5. All four pairs were baseline- and dose-eligible in this run.

| Pair | Context A ↔ B | Minimum D | Primary | D > 0.05 sensitivity | Secondary qualified; donorward; exceeds control |
|---|---|---:|---|---|---|
| Discovery | doctor/treated/patients ↔ farmer/planted/seeds | −0.003423 | fail | fail | 4/8; 0/4; 0/4 raw, unevaluable |
| Evaluation 1 | clerk/counted/coins ↔ student/answered/questions | +0.008304 | pass | fail | 2/8; 0/2; 0/2 raw, unevaluable |
| Evaluation 2 | coach/selected/players ↔ chef/sliced/tomatoes | +0.013722 | pass | fail | 6/8; 2/6; 0/6 |
| Evaluation 3 | judge/reviewed/cases ↔ painter/painted/walls | −0.014003 | fail | fail | 6/8; 1/6; 0/6 |

The discovery failure occurs for the `never` recipient in the farmer context: the weaker class-directed opposite effect is +0.005832, below the +0.009254 absolute same-cue effect. The evaluation-3 failure occurs for the `rarely` recipient in the judge context: the weaker class-directed opposite effect is +0.064359, below the +0.078362 absolute different-same-class effect. These failures do not mean the opposite-class effects were zero or had the wrong sign. Full precision, all donors and both directions are in [recipients.csv](recipients.csv) and [effects.csv](effects.csv).

The baseline screen compares cues **within** each context; it does not by itself ensure that an opposite-class donor's clean L is on the expected side of a recipient in the other context. In this run all 48 evaluation opposite-donor clean gaps happened to agree with the chosen class direction. That is an observed property of these cases, not a guarantee of the design.

The secondary defines the donor gap as same-cue alternate-context clean L minus recipient clean L. It includes a recipient-direction only if the absolute gap is at least 0.5, then reports its signed shift toward the donor and its excess over the absolute same-class shift. A pair summary requires six of eight qualified directions, baseline eligibility and q=0.5. The pooled 3/12 and 0/12 above are descriptive counts from the two evaluable pairs; there is no secondary success threshold. Individual ineligible observations and raw counts are retained in the tables.

## Full vocabulary and active controls

The earlier magnitude finding does not imply a uniquely larger full-vocabulary movement. In all three evaluation pairs, mean full-vocabulary total variation (TV) for the same-cue alternate-context donor was below that of the matched different-same-class donor:

| Evaluation pair | Same-cue full TV | Same-class full TV | Same-cue TV excluding `any`/`some` | Same-class excluded TV |
|---|---:|---:|---:|---:|
| 1 | 0.003418 | 0.005091 | 0.003723 | 0.005558 |
| 2 | 0.005991 | 0.006131 | 0.008202 | 0.008190 |
| 3 | 0.003306 | 0.006741 | 0.004169 | 0.008355 |

Excluded TV renormalizes the remaining vocabulary and can change the comparison; it is descriptive. Across all clean variants the combined probability mass of `any` and `some` ranged from 0.0130 to 0.8761, so the remaining mass ranged from 0.1239 to 0.9870. Patched target-pair mass ranged from 0.0130 to 0.8779. The per-transfer masses and both TV measures are in [effects.csv](effects.csv).

Among 168 evaluation transfers at each site, the full-vocabulary top token changed in four `cue2` transfers and six `mask2` transfers; none changed at `object2` or `cue5`. All four cue-site changes reused the same painter/`always` recipient and changed its top token from `many` to `paintings`; the six mask-site changes involved two recipient variants. These are transfer counts, not independent cases or evidence of an improved prediction. [top-tokens.csv](top-tokens.csv) records every comparison. The two-token contrast is not equivalent to the top prediction or a human acceptability judgment.

Object and mask patches were active: across evaluation transfers their signed L shifts ranged from −0.04729 to +0.02876 at `object2` and −0.10986 to +0.11774 at `mask2`; `cue2` ranged from −0.18888 to +0.12227. All final-block `cue5` transfers preserved the entire clean mask-logit vector. Clean and repeat forwards used passive capture hooks, not a separate hook-free baseline. Equal input norms do not imply equal directions or equal functional effects.

## Evidence and limits

The fixed protocol and public cases are [protocol.md](protocol.md) and [manifest.json](manifest.json). The one discovery and three evaluation pairs use a shared template and change role, verb and object together. Their texts were visible during design and are not known absent from pretraining. The cue groups also differ in lexical, frequency and temporal meaning: `rarely` is scalar rather than equivalent to explicit negation, and `any` can have free-choice uses. No isolated word, syntax, cue representation, population rate or general negation mechanism follows from these data.

The directly waited CPU process exited 0. Its closed record contains 1,088 forwards and 1,096 events: 32 clean, 128 self-patch, 896 transfer, 32 repeat, and four each of baseline and radius. Requested q was 0.5 for every pair; minimum natural donor distances were 0.862514–1.108775, requested fractions 0.025159–0.579701, and achieved norms 0.499999931–0.500000062 (maximum relative error 1.38×10⁻⁷). Independent full-witness and score/TV replays agree on all cases and outcomes; a further independent stream check matches all top-token and compact projection rows. Their scope is consistency of the saved evidence, not independent attestation that execution occurred.

The executed source is **[91f9960e419cf20df9df156a6ebc1980d1b6a864](https://github.com/EastonSell/model-looking-glass/tree/91f9960e419cf20df9df156a6ebc1980d1b6a864)**, fixed in [PR #695](https://github.com/EastonSell/model-looking-glass/pull/695). Its later merge at `5060fc419610ca7f0e2c8ec5095fb5ce93cc3531` has a different `looking_glass/intervention_plan.py`; use the executed head for exact source reproduction. The reviewer reports the separately challenged full-witness reader was publicly fixed before the first model forward in [PR #696](https://github.com/EastonSell/model-looking-glass/pull/696); the independently challenged score/TV reader was publicly fixed at [eefbac57](https://github.com/EastonSell/model-looking-glass/blob/eefbac577e06c6c8afb1746a8e7b87601b604470/tools/audit_bert_cue_context_signed_scores.py) before the operator launched the owner. These timing statements are operator/reviewer observations, not independent execution attestations.

The complete 215,157,903-byte gzip journal is retained locally with SHA-256 `a6c586dd9a8f125f60aa4003a8d8cbeac9cfae0f7862332412b5dde4a6417909`; it is omitted from this repository copy and has no public download location. The included compact projection was derived after the outcome from that journal. It retains selected `any`/`some` float32 pairs and copied TV/mass values; by itself it cannot authenticate the derivation or reconstruct full-vocabulary TV, full logits, intervention geometry, dose, landing or execution. The complete journal is required for those independent checks. Local journal location and replay details are supplied with the local user-facing bundle.

The five saved-result tables can be regenerated without a model, using a new output directory:

```text
python -I -B make-tables.py manifest.json result.json NEW_TABLE_DIRECTORY
```

With the pinned vocabulary and complete journal available, the top-token table can be regenerated in another new directory:

```text
python -I -B make-top-tokens.py manifest.json result.json FULL_JOURNAL vocab.txt NEW_TOP_DIRECTORY
```

The full-witness and score/TV readers accept the closed journal, [result.json](result.json) and [started.json](started.json); the compact projection reader operates without the journal. Their exact fixed-hash invocations are in [REPLAY.md](REPLAY.md). [inventory.json](inventory.json) records every included file's byte length and SHA-256.
