# Two-dose cue study: incomplete CPU attempt

The prospective comparison stopped at its disk-headroom guard. It did **not**
complete the prespecified study, so there is no overall pass/fail result or
paired three-triple estimate. The closed gzip prefix contains 509 records,
including 499 saved model-forward witnesses. The process was waited and exited
1. The guard fired while saving one additional returned transfer; that
forward's witness was not retained. No `result.json` was produced and the
owner's final whole-model state and runtime checks were not reached.

The [fixed protocol](protocol.md) compared requested activation-patch norms
0.5 and 0.7 at the same sites and with the same within-class,
same-recipient controls. A cue target passed only if the weaker signed
opposite-class effect exceeded the absolute within-class effect by **more
than** 0.05; the worst of the four target margins is `M`. A baseline required
all four negative-minus-affirmative clean `any`−`some` logit gaps to exceed
0.5. Two of three fresh eligible triples per cue set were needed for the
predeclared cell and paired summaries. The two completed fresh cells below
share one triple, so neither summary is available.

| Completed case | Exposure | Cue set | Baseline and both doses | M at 0.5 | M at 0.7 | Limiting target |
| --- | --- | --- | --- | ---: | ---: | --- |
| gardener / watered / plants | Exposed calibration | prior | eligible | 0.00311565 | 0.00693989 | `rarely` |
| gardener / watered / plants | Exposed calibration | new | eligible | 0.01259613 | 0.01796913 | `not` |
| chef / prepared / meals | Fresh evaluation | prior | eligible | 0.03306103 | 0.04924583 | `never` |
| chef / prepared / meals | Fresh evaluation | new | eligible | 0.01302242 | 0.01843452 | `not` |

Each completed cell's worst-target margin rose at the larger dose, but all
eight cell-dose outcomes failed the strict `M > 0.05` rule. In particular,
the completed fresh prior-cue value at 0.7 is close to the cutoff; the
machine-readable audit retains full precision and all four target margins,
individual donor and active-control effects, clean probabilities and
top-five token IDs. These are descriptive observations on one fresh lexical
triple, not evidence of a general dose response or of the original study's
aggregate outcome. The exposed calibration cells are excluded from fresh
denominators.

The `farmer / harvested / crops` prior-cue cell reached its clean baseline and
both dose screens, then stopped during the 0.5 cue-site transfers; it has no
complete specificity outcome. No records for its new-cue cell or either
`pilot / inspected / instruments` cell appear in the retained journal; the
operator reports stopping during the preceding transfer. The journal retains
the available
earlier transfers, including their controls, without imputing the missing
records. The complete input manifest was fixed and public before the run;
the 20 variants in the two calibration, two completed fresh and one partial
cells have saved forward witnesses. The named-public-input comparison is
limited to its listed snapshots and says nothing about pretraining or
unknown/private inputs.

The independently reviewed full-study reader's reviewer reports that its
bytes were fixed before that reviewer inspected numerical outcomes; they were
published in [PR #604](https://github.com/EastonSell/model-looking-glass/pull/604)
after the attempt stopped. The reader requires a complete result and cannot
by itself certify this attempt. A
separate **post-stop, related-code** prefix reader checked every saved event,
including available full-logit self/repeat and final-cue controls, selected
landing and untouched vectors, no-amplification and realized-dose geometry.
Across the four completed cells the maximum relative norm error was below
`1.091e-7`; equal input norms do not establish equal functional impact.
Clean and repeat runs used passive capture hooks. Object and mask sites are
active comparators, not null controls. The separate scalar replay checks saved
selected logits and decisions. Neither reader attests the model execution.

The predeclared calibration comparison found exact equality for all 144
corresponding clean, self, 0.5-transfer and repeat records against the earlier
factorial journal, including full mask-logit bits and saved state/landing
vectors. This is a descriptive check on already exposed inputs. It does not
repair the missing fresh records or prove cross-runtime reproducibility.

Recorded execution source was
[`b75e5bcc3a866a9af56ac1ac3ba54c6abe1691a4`](https://github.com/EastonSell/model-looking-glass/tree/b75e5bcc3a866a9af56ac1ac3ba54c6abe1691a4).
The source PR merged as
[`4d2031b8`](https://github.com/EastonSell/model-looking-glass/commit/4d2031b8d23b864d5dbeee753d8db0f37607d549),
but main contains a different shared intervention helper; it is not an
identical execution tree. The operator reports that the source was public
before the first follow-up forward. The no-forward preflight checked the 17
bound files, cached assets, package versions and all 32 tokenizer rows; it did
not import Torch or run a model. These records assume a trusted quiescent
checkout, interpreter and dependencies; hashes do not attest execution.

The complete 98,085,910-byte compressed prefix is retained in the local
user-facing output `bert-cue-dose-response-cpu-v1-20261029`; it is omitted
from this repository copy and has no public download location. The compact
target-logit subset supports an independent selected-score replay, but its
hash alone does not authenticate its derivation or verify the full logits,
landing, dose or execution. The raw failure traceback is withheld because it
contains local paths; [failure-summary.json](failure-summary.json) is a
path-free operator projection and its raw-file hash is a commitment only.
The compact [inventory](inventory.json) indexes the public files. The local
output also contains the full journal and raw failure record; its separate
`local-inventory.json` indexes those additional local files.

From the compact bundle, the selected-score replay needs only Python's
standard library:

```text
python -I -B audit_subset.py manifest.json score-subset.jsonl.gz --expected-subset-sha256 cc2c40a30b55d9c3e150405a9b5c100c13cea0e6da99299d8a620c7a9708da13 --output replay.json
```

`replay.json` must be a new path; its bytes match
[score-report.json](score-report.json). With the retained
full journal and original path-bearing failure file available locally, the
post-stop full-witness prefix check can also be run from the local bundle:

```text
python -I -B audit_prefix.py --reader audit_full.py --manifest manifest.json --events events.jsonl.gz --started started.json --failure failure.json --receipt controller-exit.json
```

Its output matches [prefix-audit.json](prefix-audit.json). The
post-interruption [subset derivation](derive_subset.py) requires the retained
original journal and records; it cannot be reproduced from the public subset
alone. The small [calibration comparator](compare_calibration.py) likewise
requires both local full journals.

This is one fixed clause placement with public, nonblind cases and cue words
that differ in lexical, frequency and temporal meaning. The `any`−`some`
contrast is not a grammaticality judgment or a test of general negation
understanding. The 0.7 dose remains below the earlier studies' smallest
observed radius (about 0.7678), and these observations cannot identify why
those studies differed or isolate syntax. No linear or monotonic dose curve,
population inference, or pretraining-disjointness follows.
