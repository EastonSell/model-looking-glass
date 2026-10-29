# DistilBERT cue set × clause placement: fixed-dose CPU result

All four cells **failed** the prespecified specificity rule. Each had three
eligible lexical triples and **0/3** passing cases. Across the 12 cases,
13 of 48 individual cue-target checks passed, but no case passed all four.
Eleven of the 12 worst-target margins were positive and one was negative:
failure means the required excess over the same-recipient control did not
clear 0.05, not that transfers had no directional effect.

This is a fixed `any`-versus-`some` masked-logit experiment in one model. It
does not establish general negation understanding or identify a single cause
of the difference between the two earlier studies.

## The fixed comparison

The [protocol](protocol.md) crossed two cue sets with two clause placements,
using the same three role/verb/object triples in every cell:

* Prior cues: negative/low-frequency `never` and `rarely`, versus affirmative
  `always` and `often`.
* New cues: negative/minimizing `not` and `hardly`, versus affirmative
  `really` and `certainly`.
* Main: `The {role} has {cue} {verb} [MASK] of the {object}.`
* Subject relative: `The {role} who has {cue} {verb} [MASK] of the {object} has signed the form.`

The triples were teacher/graded/essays, gardener/watered/plants and
mechanic/repaired/engines. According to the operator's chronology, all 48
variant rows were publicly fixed before this study's first model forward.
They are three paired lexical triples, not
12 independent examples or a sample known to be absent from pretraining. The
main placement resembles the first study's main-clause configuration but is
not its exact template: common `has` was added so both cue sets fit
grammatically, changing aspect. The frames also differ in prefix, position,
length and discourse; cues differ in temporal, frequency and intensity
semantics. The comparisons cannot isolate syntax from these differences.

For `L = logit(any) - logit(some)`, a baseline was eligible only when all four
negative-minus-affirmative clean gaps exceeded 0.5 strictly. Dose eligibility
required all 24 within-cell cue-pair distances across the four sites to be at
least the fixed requested radius 0.5. Every case passed both screens; the 48
clean gaps ranged from +2.9910 to +7.7504. No cases were omitted.

At the block-2 cue position, each target's margin is its weaker signed
opposite-class donor effect minus the absolute effect of the other
same-class donor into the same recipient, at the same requested radius. The
case score `M` is the minimum of its four target margins; a case passes only
if `M > 0.05` strictly. A cell required at least two eligible and at least two
passing triples. All four cells were evaluable and failed:

| Cue set | Main | Subject relative |
| --- | ---: | ---: |
| Prior | 0/3 | 0/3 |
| New | 0/3 | 0/3 |

All case scores and their limiting targets are below. Values are rounded;
decisions use full precision. Exactly tied limiting targets would all be
reported, and the limiting target can change between cells.

| Triple | Prior / main | New / main | Prior / relative | New / relative |
| --- | ---: | ---: | ---: | ---: |
| teacher / graded / essays | +0.02453 (`always`) | +0.01746 (`not`) | +0.01310 (`always`) | +0.00712 (`not`) |
| gardener / watered / plants | +0.00312 (`rarely`) | +0.01260 (`not`) | −0.00220 (`rarely`) | +0.00751 (`hardly`) |
| mechanic / repaired / engines | +0.02552 (`rarely`) | +0.00865 (`not`) | +0.02798 (`rarely`) | +0.00154 (`not`) |

The complete 48 clean baselines and vocabulary top-five predictions, 576
donor effects, 192 site/target checks and per-case eligibility are in
[baselines.csv](baselines.csv), [effects.csv](effects.csv),
[target_checks.csv](target_checks.csv) and [cases.csv](cases.csv).

## Paired differences and interpretation

The prespecified paired score differences are descriptive. For each triple,
the cue-set difference is `M(new) - M(prior)` within a frame; the frame
difference is `M(relative) - M(main)` within a cue set. All three triples were
eligible for each contrast and for the interaction, which is the relative
cue-set difference minus the main cue-set difference. For exact binary64
replay, the interaction uses one four-term `fsum`, as specified in the
protocol, rather than subtracting two separately rounded differences.

| Contrast | Mean | Individual triple differences |
| --- | ---: | --- |
| New − prior, main | −0.00482 | −0.00707, +0.00948, −0.01688 |
| New − prior, relative | −0.00757 | −0.00598, +0.00971, −0.02644 |
| Relative − main, prior | −0.00476 | −0.01143, −0.00531, +0.00246 |
| Relative − main, new | −0.00751 | −0.01033, −0.00509, −0.00711 |
| Interaction | −0.00275 | +0.00110, +0.00023, −0.00956 |

These are not significance tests or population estimates. Since `M` is a
minimum over different words and its limiting target changes, its interaction
is not an effect on a fixed target or an abstract polarity parameter. Screening
also makes the estimands conditional on model behavior. Full-precision paired
values and eligible denominators are in [contrasts.csv](contrasts.csv).

The earlier [cue-class study](https://github.com/EastonSell/model-looking-glass/blob/a13a2eab68d02cbf93f43af9ad01559364d129fd/results/bert-cue-class-cpu-v1-20261029/README.md)
passed 7/9 cases; the later [held-out-frame study](https://github.com/EastonSell/model-looking-glass/blob/975d55aea042912e88cee86b13738d490fea18b6/results/bert-cue-heldout-frames-cpu-v1-20261029/README.md)
passed 0/6. Here both cue sets fail both placements under one common 0.5 norm.
That norm is smaller than every minimum radius observed in those earlier
studies (their combined minimum was 0.7678), and may reduce effects below the
fixed threshold. The present result does not disentangle how much the earlier
difference arose from cue, frame, lexical choices, dose or other factors.

The two-token contrast is not a full-vocabulary winner or acceptability
judgment. For example, in the gardener subject-relative case with `rarely`,
`one` was the top mask token even though `any` had probability 0.30568 versus
0.06325 for `some`. `any` can have free-choice readings; `rarely` and `hardly`
are scalar and not interchangeable with explicit negation.

## Delivery, controls and evidence

All 12 cases used requested norm 0.5. The minimum natural pair/site distance
per cell ranged from 0.69426 to 1.26289; requested fractions ranged from
0.02695 to 0.72020. The full-vector audit checked all 576 transfer landings and
untouched-token witnesses, 192 same-forward self patches, 48 final clean
repeats and 144 final-block cue transfers. Self/repeat and final-cue
full-logit controls passed, and every realized dose satisfied the 0.001
relative tolerance. Clean and repeat forwards used passive capture hooks;
there was no separately hook-free baseline.

The object and mask sites were active comparators: individual changes in `L`
ranged −0.04002 to +0.03766 at the object and −0.11962 to +0.12950 at the
mask. Cue-site effects ranged −0.17487 to +0.14191. Final-block cue transfers
left the complete mask-logit vector unchanged. Equal input norms do not make
functional impacts equal; the comparisons do not identify a unique site.

The operator waited for the direct CPU process and observed exit 0; the owner
reported 864 forwards and 888 saved events. The exact recorded execution source
is [commit 2889532b](https://github.com/EastonSell/model-looking-glass/tree/2889532b4b657e3d6f1c00566f8ed5df7f2b2e33).
The source PR merged at `ba8d34b9`, but concurrent main changes made
`looking_glass/intervention_plan.py` differ from the executed version; use
the recorded commit for exact source reproduction. The no-forward preflight
checked 15 source files, five pinned assets, installed package versions and
all 48 tokenized rows without importing Torch or loading the model.

The [full-witness reader](https://github.com/EastonSell/model-looking-glass/pull/575)
was independently challenged and publicly fixed before real outcome
inspection. A separate scalar reader independently reproduced the screens,
case decisions and paired contrasts. These check saved bytes and recorded
identities; neither independently attests execution or pretraining exposure.
The source, interpreter and dependencies were trusted to remain quiescent.
The public row comparison covers ten named snapshots only and does not clear
private or unknown exposures.

After outcome inspection, a separate reader derived an operator-supplied
[score subset](score-subset.jsonl.gz) containing event identities and exact
float32 `any`/`some` logits. The subset reader replays the fixed screen,
decisions and paired contrasts without the complete journal. The subset alone
does not authenticate its extraction from the original journal or verify full
logits, dose, landing, unchanged tokens or execution. The public subset replay
uses only Python's standard library:

```text
python -B audit_subset.py manifest.json score-subset.jsonl.gz result.json --expected-source 2889532b4b657e3d6f1c00566f8ed5df7f2b2e33 --expected-result-sha256 d3276451d0ee6793acd5f84063a65cf00ac6f511a0e0bad0048fa64e1da30caf --expected-subset-sha256 b8df06d2ab871523c1b5508ff8f42aaf965378808efc6d67b8c45caff1d69256 --expected-original-journal-sha256 667004f67abfae1fc97f66dd6a1061e352fc03a6d90c8ad47a49e862903520d9
```

The complete journal is omitted from this repository copy and has no public
download. With the locally retained full journal beside these files, Python's
standard library can replay both saved-data audits:

```text
python -B audit_full.py manifest.json events.jsonl.gz result.json --started started.json --expected-started-sha256 6c91962581e9cc2427427a64357dba838029336b093a39a45b07f44487e52b96 --expected-result-sha256 d3276451d0ee6793acd5f84063a65cf00ac6f511a0e0bad0048fa64e1da30caf --expected-source 2889532b4b657e3d6f1c00566f8ed5df7f2b2e33
python -B review_bert_cue_factorial_saved_scores.py manifest.json events.jsonl.gz result.json --expected-result-sha256 d3276451d0ee6793acd5f84063a65cf00ac6f511a0e0bad0048fa64e1da30caf --expected-source 2889532b4b657e3d6f1c00566f8ed5df7f2b2e33
python -B derive_subset.py manifest.json events.jsonl.gz rebuilt-subset.jsonl.gz --expected-manifest-sha256 b91eba1e0a52f6a183a59b0bfbb365dd54d67341bd74a4fb56325e72a7f45d4f --expected-journal-sha256 667004f67abfae1fc97f66dd6a1061e352fc03a6d90c8ad47a49e862903520d9
```

The checkpoint is not included. [inventory.json](inventory.json) binds the
included evidence; [make_tables.py](make_tables.py) rebuilds the CSVs from
the saved result, manifest and separately supplied pinned vocabulary. The
178,767,395-byte journal is retained in the local user-facing directory and
omitted here. Its hash alone does not permit a full-vector replay.
