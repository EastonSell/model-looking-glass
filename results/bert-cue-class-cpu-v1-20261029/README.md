# DistilBERT cue-class specificity: CPU follow-up

The fixed specificity criterion **passed** on the nine public evaluation
contexts: all three main-frame contexts and two of three in each of the
embedded and relative frames passed. All nine met the baseline and dose
screens. Two evaluation contexts failed the four-target rule, and one of the
three development contexts also failed. These are retained below.

This is a conditional result about the model's masked `any` versus `some`
score under equal-requested-norm cue-state replacements. The four cues differ
in frequency, temporal meaning and lexical identity, so the finding does not
establish general negation understanding or a uniquely localized mechanism.

## Fixed comparison

For each context, the study compared `never` and `rarely` (the operational
negative/low-frequency class) with `always` and `often` (the high-frequency
affirmative class). `rarely` is not explicit sentential negation. Let `L` be
the `any` logit minus the `some` logit. The baseline screen required all four
clean `L(negative) - L(affirmative)` gaps to be strictly greater than 0.5.

At the block-2 cue position, each target's score was the smaller of the two
opposite-class effects, signed in the predicted direction, minus the absolute
effect of the other cue in the target's own class. Every target had to exceed
0.05 logits strictly for its context to pass. The same-class donor was applied
to the same recipient at the same requested radius, so a comparable within-
class effect could make the target fail. A frame needed at least two eligible
and two passing evaluation contexts; fewer than two eligible would have made
the primary criterion unevaluable. No such ineligibility occurred.

The [protocol](protocol.md) fixed three development and nine evaluation
contexts before this follow-up's model forwards. The designated evaluation
cases were public, not hidden or known to be absent from pretraining. The
frames were a main-clause partitive, an embedded
complement with a nonpartitive masked noun phrase, and a main clause following
a subject-modifying relative clause. The last cue and mask are in the main
clause. Frame comparisons also change lexical items, partitivity, length and
token positions; their differences cannot be assigned to syntax alone. Each
frame had one development context using the same template, so the evaluation
does not hold out entire syntactic frames from development. All cases and
decisions were fixed before follow-up outputs.

## All cases

The table gives the smallest of four clean baseline gaps and the four
cue-position specificity margins. Every case was baseline- and dose-eligible.
The five-decimal display is rounded; decisions use the saved full-precision
values and the strict threshold.

| Case | Split | Role / verb / object | Minimum clean gap | never | rarely | always | often | Pass |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| dev-main | development | gardener / watered / plants | 3.98886 | 0.05634 | -0.00552 | 0.07183 | 0.05679 | no |
| dev-embedded | development | nurse / checked / records | 2.90206 | 0.13248 | 0.06098 | 0.12541 | 0.11595 | yes |
| dev-relative | development | baker / tasted / cakes | 7.53358 | 0.07662 | 0.06710 | 0.28053 | 0.23679 | yes |
| eval-main-01 | evaluation | mechanic / repaired / engines | 4.98051 | 0.14850 | 0.08446 | 0.15699 | 0.12873 | yes |
| eval-main-02 | evaluation | teacher / graded / papers | 3.41776 | 0.28104 | 0.07975 | 0.05480 | 0.09352 | yes |
| eval-main-03 | evaluation | librarian / organized / books | 4.16530 | 0.09866 | 0.10271 | 0.10807 | 0.10401 | yes |
| eval-embedded-01 | evaluation | doctor / reviewed / charts | 3.72498 | 0.05020 | -0.00943 | 0.15743 | 0.14724 | no |
| eval-embedded-02 | evaluation | farmer / collected / eggs | 5.81683 | 0.09380 | 0.09920 | 0.16417 | 0.13003 | yes |
| eval-embedded-03 | evaluation | writer / sent / letters | 6.40805 | 0.08383 | 0.09359 | 0.30148 | 0.21649 | yes |
| eval-relative-01 | evaluation | driver / inspected / buses | 3.66379 | 0.09825 | 0.03388 | 0.04827 | 0.08184 | no |
| eval-relative-02 | evaluation | chef / prepared / meals | 6.54353 | 0.09300 | 0.13426 | 0.12990 | 0.15099 | yes |
| eval-relative-03 | evaluation | artist / copied / sketches | 5.68946 | 0.19858 | 0.17091 | 0.09926 | 0.11846 | yes |

The embedded failure provides a direct counterexample to uniform specificity:
for the `rarely` recipient, the weaker signed opposite-class effect was
+0.16846 while the within-class `never` donor's absolute effect was +0.17789,
giving a margin of -0.00943. For the relative failure, the `rarely` and
`always` margins were +0.03388 and +0.04827, both below the strict +0.05
threshold. The embedded case's `never` margin was +0.0502045, just above the
cutoff, even though that case failed on `rarely`. Exact cross-runtime
reproduction of a near-threshold margin is not established. The full
per-donor effects and per-target margins at all four sites
are in [effects.csv](effects.csv) and [target_checks.csv](target_checks.csv).

The clean masked prediction was not always `any` even in the low-frequency
variants. For example, the `never` and `rarely` variants of the failed
embedded evaluation case ranked `chart` first. [baselines.csv](baselines.csv)
records each clean margin, `any` and `some` probabilities and the full-vocabulary
top five; [result.json](result.json) retains the token IDs, scores, all baseline
gaps and intervention effects. A two-token contrast is not a top-one or
acceptability judgment.

## Delivery and active controls

The common radius for each context was the minimum of the 24 specified
pairwise distances across four sites. Requested radii ranged from
0.800859 to 1.389140, and requested fractions of the natural donor distance
from 0.056604 to 1.0. The largest realized relative dose error across 576
transfers was 5.16e-8, below the fixed 0.001 tolerance. Equal input norms do
not ensure equal functional impact, and a minimum across sites can make some
replacements small relative to their natural displacement.

The independent full-vector replay checked all 192 self patches, 48 final
clean repeats, 576 transfer landings and unchanged-token witnesses, and 144
final-block cue transfers. All corresponding full-logit controls passed.
Clean and final-repeat forwards used passive capture hooks at the four sites;
they do not provide a separately hook-free baseline.
Applying the same specificity formula descriptively at the block-2 object and
mask positions gave 0/9 passing evaluation contexts at each site, but those
are active controls: individual evaluation effects ranged from -0.0971 to
+0.0874 logits at the object and -0.3039 to +0.3383 at the mask. Final-block
cue transfers preserved the complete mask-logit vector. These site summaries
are descriptive and do not establish unique localization.

## Evidence and scope

The operator observed a direct CPU process exit of 0; the owner reported 864
forwards and 888 saved events. According to the operator's observed chronology,
the [source and fixed protocol](https://github.com/EastonSell/model-looking-glass/pull/508)
were publicly available before the first follow-up forward. The recorded
execution source was `84f15048a22c094794a86a6c41946e276df876a4`; the 13
bound files match the merged source, while the complete trees differ. The
no-forward preflight blocked model forward, selected network calls and CUDA
initialization, and checked the pinned CPU checkpoint and all 48 tokenizer
rows. The actual owner reported no CUDA initialization and matching model
state digests before and after the run.

Two independent standard-library readers replayed the closed saved evidence.
The full reader was locally fixed and challenged on fabricated data before
outcome inspection; its [public readback](https://github.com/EastonSell/model-looking-glass/pull/513)
occurred after the owner started and before that reader inspected the study
output. It checks full mask logits, all intervention vectors, delivery and
controls; the separate scalar reader checks action order, target logits,
baseline and decisions. These are saved-record consistency checks, not
independent execution or pretraining-exposure attestations. The source
checkout, interpreter and installed dependencies are trusted to remain
quiescent during execution.

The complete journal is omitted from this repository copy, so the full replay
cannot be run from these files alone. It is retained in a local user-facing
bundle; there is currently no public journal download. With that journal
placed beside these files, Python's standard library can run:

```text
python -B audit_full.py manifest.json events.jsonl.gz result.json --started started.json --expected-started-sha256 b11a1d6e91c67071119ba466f31a1dfbab8bebd5a645684b82efb6775a93307d --expected-result-sha256 40c71340878c8897aae105c1efc3eaf347864f4d90f97617b9f7af0b8e867575 --expected-source 84f15048a22c094794a86a6c41946e276df876a4
python -B audit_scores.py manifest.json events.jsonl.gz result.json --expected-result-sha256 40c71340878c8897aae105c1efc3eaf347864f4d90f97617b9f7af0b8e867575 --expected-source 84f15048a22c094794a86a6c41946e276df876a4
```

The checkpoint is not included. [inventory.json](inventory.json) binds the
included evidence; [make_tables.py](make_tables.py) reconstructs the CSVs
from the result, manifest and a separately supplied pinned `vocab.txt`. The
172,274,338-byte journal is present in the local user-facing bundle and
omitted here. Its hash is recorded in the result and inventory metadata;
a hash alone does not make the full-vector audit reproducible.

The cases were selected after the earlier cue study and after a preliminary
full-vocabulary analysis of that study had been received. A tokenizer-only
candidate replacement was recorded before follow-up forwards. The exact-row
check against seven selected public BERT inventories found no overlap; it does
not clear unknown or private exposures. The three related contexts per frame
are convenience examples, not independent draws, and the baseline screen
conditions the success rate. `any` can have free-choice readings, and the cue,
lexical, temporal and frame differences remain confounds. Prior motivation
includes [Warstadt et al. (2019)](https://aclanthology.org/D19-1286/),
[DeCarlo et al. (2023)](https://aclanthology.org/2023.blackboxnlp-1.25/),
and [Kletz et al. (2023)](https://aclanthology.org/2023.nodalida-1.54/).
