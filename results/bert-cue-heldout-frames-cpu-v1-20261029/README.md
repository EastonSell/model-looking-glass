# DistilBERT held-out cue frames: negative CPU result

The fixed specificity criterion **failed**: none of the three subject-relative
contexts or three causal-adjunct contexts passed all four cue-target checks.
All six contexts passed the clean-baseline and dose screens, so the criterion
was evaluable. Fourteen of the 24 individual cue-target checks passed; every
context failed at least the `not` target. This is a failure of the prespecified
excess over a within-class control, not evidence that cue transfer was zero.

The test concerns one model's `any` versus `some` masked-token score under
these interventions. The cues differ in meaning and distribution, so the
result does not establish general negation understanding or its absence.

## Fixed design and decision

The [protocol](protocol.md) fixed two new templates before the follow-up's
model forwards. In one, both cue and mask occur inside a subject-relative
clause; in the other, both occur in a causal adjunct:

* `The {role} who has {cue} {verb} [MASK] of the {object} has signed the form.`
* `The clerk is writing a note because the {role} has {cue} {verb} [MASK] of the {object}.`

Three role/verb/object triples were paired across both frames. Neither template
had a development example in the earlier study or this one, and there were no
new development forwards. All cases were public and fixed before outputs,
not hidden or known to be absent from pretraining. The same triples constrain
cross-frame lexical variation, but these six cells are paired rather than six
independent lexical examples. Position, prefix, length and causal/relative
meaning still differ; a frame difference cannot be attributed to syntax alone.

The negative/minimizing cues were `not` and `hardly`, and the affirmative cues
were `really` and `certainly`. Holding `has` fixed controls the auxiliary and
aspect, but does not make the cues semantically interchangeable or establish
a reduction in frequency confounding. `hardly` is a scalar minimizer, and
`really` and `certainly` differ in intensity and certainty. `any` can also
receive free-choice readings; the results are not a human acceptability test.

Let `L = logit(any) - logit(some)`. A context was baseline-eligible only if
all four clean gaps `L(not/hardly) - L(really/certainly)` were strictly
greater than 0.5. Each target's specificity margin at the block-2 cue site was
the weaker of its two opposite-class donor effects, signed in the predicted
direction, minus the absolute effect of the other same-class donor into that
same recipient at the same requested radius. All four target margins had to
exceed 0.05 logits strictly. At least two of three contexts in each frame had
to be eligible and pass; fewer than two eligible in either frame would have
made the aggregate unevaluable. No contexts were excluded.

| Case | Role / verb / object | Smallest clean gap | not | hardly | really | certainly | Pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| subject-relative 01 | inspector / examined / packages | 3.81117 | 0.00325 | 0.05440 | 0.05438 | 0.11456 | no |
| subject-relative 02 | assistant / sorted / reports | 3.23760 | 0.01772 | 0.01637 | 0.04222 | 0.09235 | no |
| subject-relative 03 | worker / cleaned / tools | 4.65340 | -0.01285 | 0.02482 | 0.09827 | 0.17822 | no |
| causal-adjunct 01 | inspector / examined / packages | 4.90485 | 0.02398 | 0.02566 | 0.25073 | 0.27708 | no |
| causal-adjunct 02 | assistant / sorted / reports | 4.58981 | 0.04792 | 0.06469 | 0.10112 | 0.11333 | no |
| causal-adjunct 03 | worker / cleaned / tools | 5.58796 | 0.03505 | 0.07370 | 0.16692 | 0.18291 | no |

The four cue columns are target specificity margins. Displayed values are
rounded; the saved decisions use full precision and strict thresholds. All
24 clean cross-class gaps ranged from +3.23760 to +6.38490. Complete per-case
baselines, all 288 donor-to-target effects, and all 96 target/site margins are
in [baselines.csv](baselines.csv), [effects.csv](effects.csv) and
[target_checks.csv](target_checks.csv).

## What failed, and what did not

For the `not` recipient, the weaker signed opposite-class donor effect was
positive in every context, but did not exceed the absolute `hardly`-to-`not`
effect by the required 0.05:

| Frame / triple | Weaker signed opposite effect | `hardly`-to-`not` effect | Margin |
| --- | ---: | ---: | ---: |
| subject-relative / inspector | +0.02828 | +0.02503 | +0.00325 |
| subject-relative / assistant | +0.02093 | +0.00321 | +0.01772 |
| subject-relative / worker | +0.01737 | +0.03021 | -0.01285 |
| causal-adjunct / inspector | +0.03253 | +0.00855 | +0.02398 |
| causal-adjunct / assistant | +0.05464 | +0.00671 | +0.04792 |
| causal-adjunct / worker | +0.05523 | +0.02019 | +0.03505 |

In the subject-relative worker case, the within-class effect was larger than
the weaker opposite-class effect. In the others, the opposite-class effect
was larger but did not clear the fixed margin. The prior [cue-class study](https://github.com/EastonSell/model-looking-glass/blob/a13a2eab68d02cbf93f43af9ad01559364d129fd/results/bert-cue-class-cpu-v1-20261029/README.md)
passed 7/9 evaluation contexts; the new 0/6 outcome is not a controlled estimate
of a frame effect because the cues, lexical triples and frames changed together.

The two-token contrast was not always the full-vocabulary top prediction.
For the subject-relative assistant case with `not`, `out` ranked first while
`any` had probability 0.25462 and `some` 0.01026. The complete clean top five
and both probabilities for every cue are in [baselines.csv](baselines.csv).

## Delivery and controls

Each context used the minimum of 24 specified cross-cue distances across the
four sites as its requested radius. Radii were 0.767801–0.999231; requested
fractions of natural donor distances were 0.041968–1.0. Across 288 transfers,
the maximum realized relative norm error was 7.47e-8, below the 0.001
tolerance. Equal input norms do not ensure equal functional impact.

The independent full-vector replay checked 96 self patches, 24 final clean
repeats, all 288 transfer landings and unchanged-token witnesses, and 72
final-block cue transfers. The corresponding full-logit controls passed.
Clean and repeat forwards used passive capture hooks; these checks do not
include a separately hook-free baseline.

At the contextual object and mask sites, applying the same four-target
formula descriptively also gave 0/6 passing contexts. Those are active
comparators: individual raw effects ranged from -0.05873 to +0.06900 logits
at the object and -0.09948 to +0.30027 at the mask. The final-block cue
transfers preserved the whole mask-logit vector. These observations do not
establish a unique location or the absence of effects at other doses.

## Evidence and limits

The operator directly observed the CPU process exit 0; the owner reported 432
forwards and 444 saved events. According to the operator's chronology, the
[protocol and source](https://github.com/EastonSell/model-looking-glass/pull/541)
were publicly fixed before the first follow-up forward. The recorded source was
`7eec34c31dbeb16af00d9d80301525587db8dd05`; its 15 bound files match the
merged source, though the whole trees differ. The no-forward preflight blocked
model forward, selected network calls and CUDA initialization, and checked
the pinned CPU checkpoint and all 24 tokenizer rows. The owner reported no
CUDA initialization and matching model-state digests before and after.

Two independent standard-library readers replayed the closed evidence. The
[full reader](https://github.com/EastonSell/model-looking-glass/pull/548) was
fixed and challenged on fabricated records before real-output inspection; it
checks complete mask logits, all intervention vectors, delivery and controls.
The scalar reader separately checks event order, target logits, baseline and
decisions. These are saved-record consistency checks, not independent execution
or pretraining-exposure attestations. The source checkout, interpreter and
installed dependencies are trusted to remain quiescent during execution.

After outcome inspection, a separate reader derived the compact
[score subset](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/bert-cue-heldout-frames-cpu-v1-20261029/score-subset.jsonl.gz): for each forward it retains only the
exact float32 `any` and `some` logit bytes and the fields needed to replay the
fixed decision. The public subset reader can check that arithmetic and action
order without the full journal. The subset alone does not authenticate its
derivation from the original journal or verify full-vocabulary logits, dose,
landing, untouched-token witnesses or model execution.

The subset replay works from either bundle with Python's standard library:

```text
python -B audit_subset.py manifest.json score-subset.jsonl.gz result.json --expected-source 7eec34c31dbeb16af00d9d80301525587db8dd05 --expected-result-sha256 a9b634310efaa24821a4ba79f9a47a036f1066a085ff89ea162e970306ab6e76 --expected-subset-sha256 bc459923f35c22d639636a2bdf7a39888e220c05e8502bd1f25ae29aed188300 --expected-original-journal-sha256 345463b525afaed573656bbd9fbfa72afa8225437a461957d24169a67c5e262d
```

The complete journal is omitted from this repository copy and has no public
download. With the locally retained full journal beside these files, Python's
standard library can replay the full saved evidence:

```text
python -B audit_full.py manifest.json events.jsonl.gz result.json --started started.json --expected-started-sha256 e5608ff2ee665ac5a28db31eedca4ac90c1cd2b8658f0c3c3666d581621ee515 --expected-result-sha256 a9b634310efaa24821a4ba79f9a47a036f1066a085ff89ea162e970306ab6e76 --expected-source 7eec34c31dbeb16af00d9d80301525587db8dd05
python -B audit_scores.py manifest.json events.jsonl.gz result.json --expected-result-sha256 a9b634310efaa24821a4ba79f9a47a036f1066a085ff89ea162e970306ab6e76 --expected-source 7eec34c31dbeb16af00d9d80301525587db8dd05
python -B derive_subset.py manifest.json events.jsonl.gz rebuilt-subset.jsonl.gz --expected-manifest-sha256 7a10dbcb598e48c2e8f56e2b75d8c03188ec04ca7a21ac071aea4b1e8294162c --expected-journal-sha256 345463b525afaed573656bbd9fbfa72afa8225437a461957d24169a67c5e262d
```

The checkpoint is not included. [inventory.json](inventory.json) binds the
included evidence; [make_tables.py](make_tables.py) reconstructs the tables
from the result, manifest and a separately supplied pinned `vocab.txt`. The
97,698,957-byte journal is present in the local user-facing bundle and
omitted here. Its recorded hash does not by itself permit full-vector replay.

Selection followed observation of the earlier study's results, and the
rejected pre-forward suffix-only proposal is recorded in the manifest. An
exact and complete-prefix check against eight selected public BERT inventories
found no overlap, but does not clear unknown or private exposures. The sample
is small, paired and conditional on the clean screen. Prior work motivating
the cue and licensing limitations includes [Warstadt et al. (2019)](https://aclanthology.org/D19-1286/),
[DeCarlo et al. (2023)](https://aclanthology.org/2023.blackboxnlp-1.25/) and
[Hsieh (2012)](https://linguistics.uconn.edu/wp-content/uploads/sites/1842/2020/05/HsiehI_2012_Polarity-in-Conditionals-and-Conditional-like-Constructions.pdf).
