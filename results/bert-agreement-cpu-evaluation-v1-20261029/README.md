# DistilBERT agreement: balanced CPU evaluation

The predeclared directional and matched-layer distractor criteria both passed on this
small, public evaluation set. The candidate intervention's mean signed effect
was **+4.8537 logits**, with a positive mean in all eight families. Its mean
exceeded the block-2 distractor comparison by **+5.0179 logits**. A fixed
comparison at the **earlier block-0 subject position was larger in every
family**, averaging **+7.0311 logits**. Thus the matched distractor comparison
does not show that the candidate block uniquely carries the relevant effect.

This is one checkpoint, one template and eight dependent, cyclically paired
noun families. Positive effects can result from donor-associated lexical or
surface changes or attenuation of the target preference; they do not establish
transfer of a unique grammatical-number feature. The public evaluation cases
were designed to be held out from local site selection and were not observed
in the cited project records before freeze. They were not blind, and uncited
local exposure, exposure elsewhere or in pretraining is unknown. Both the nouns
and preposition changed from development, so this run does not separate lexical
from template generalization.

## Frozen experiment and scope

The pinned model was `distilbert/distilbert-base-uncased` at revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`. Its model weights SHA-256 was
`5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063`.
The [frozen protocol](https://github.com/EastonSell/model-looking-glass/blob/4ee50069464228798071218b58b64221ac5960f4/docs/BERT_AGREEMENT_V1.md),
[public cases](https://github.com/EastonSell/model-looking-glass/blob/4ee50069464228798071218b58b64221ac5960f4/experiments/bert_agreement_public_cases_v1.json), and
[pre-evaluation exposure snapshot](https://github.com/EastonSell/model-looking-glass/blob/4ee50069464228798071218b58b64221ac5960f4/experiments/bert_agreement_exposures_v4.json)
specify the exact inputs, four crossed donor conditions, five sites, controls,
numerical reductions and decision rules. Source commit
[`4ee50069464228798071218b58b64221ac5960f4`](https://github.com/EastonSell/model-looking-glass/commit/4ee50069464228798071218b58b64221ac5960f4)
was observed published before the local launch and was subsequently merged in
[PR #258](https://github.com/EastonSell/model-looking-glass/pull/258). These
are local chronology observations, not an independently witnessed timestamp.

The run used one CPU owner, one intra-op and one inter-op thread, float32,
evaluation mode, eager attention and the fixed runtime/asset bindings in the
terminal. A separate local read-only preflight rehashed the five assets and
validated the 64-row manifest and exposure binding before launch. The
controller's own preflight checked committed source and bindings; the owner
then staged and verified the assets and, after model loading, checked all 96
manifest rows' tokenizations before its first forward. The completed record contains
1,120 started and completed forwards, 704 replacement-landing checks, and 384
passing identity, clean-output and delivery controls. It contains all 32
target cases; no model error was filtered out. The local attempt ledger had no
prior entry and designated this the first complete primary result; it does not
prove that the cases were never previously exposed elsewhere. All 64
evaluation input rows in this attempt are now observed in the cited evidence.

## Results

| Site | Mean signed donor-number effect (logits) | Candidate minus site |
| --- | ---: | ---: |
| Block 2, subject: candidate | +4.8537 | — |
| Block 2, distractor: prespecified comparison | -0.1641 | +5.0179 |
| Block 3, distractor: fixed comparison | -0.0631 | +4.9168 |
| Block 4, distractor: fixed comparison | -0.0149 | +4.8686 |
| Block 0, subject: fixed comparison | +7.0311 | -2.1774 |

The directional rule required a candidate mean greater than 0.1 and at least
six positive family means; it passed with eight. The separate specificity rule
required candidate minus block-2 distractor greater than 0.1; it also passed.
The larger block-0 effect is a prespecified descriptive comparison, not a
separate significance test.

The four baseline cells each had 8/8 correct preferences. Mean `is - are`
logit margins were +4.9676 for singular subject/singular distractor, +1.3098
for singular/plural, -3.7388 for plural/singular, and -5.0364 for
plural/plural. Mean candidate effects for those cells were +4.5873, +5.3087,
+6.4148 and +3.1040 logits respectively.

| Family | Candidate mean | Block-0 subject mean | A-opposite reversals | B-opposite reversals |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.7329 | 7.6412 | 2/4 | 2/4 |
| 2 | 4.3075 | 6.7956 | 2/4 | 2/4 |
| 3 | 5.3008 | 6.9803 | 2/4 | 3/4 |
| 4 | 5.8737 | 8.3388 | 3/4 | 3/4 |
| 5 | 4.5000 | 6.7554 | 2/4 | 2/4 |
| 6 | 4.4341 | 5.8813 | 2/4 | 2/4 |
| 7 | 4.3615 | 6.7545 | 2/4 | 3/4 |
| 8 | 5.3191 | 7.1015 | 2/4 | 2/4 |

A and B are the same-lemma and different-lemma donors defined by the protocol.
A reversal here means the patched and clean `is - are` margins have strictly
opposite signs; ties do not count. The totals are 17/32 for A and 19/32 for B;
neither opposite-number patch reversed any of the eight plural/plural cases
(0/16 actions). All clean margins and the 64 patched margins were nonzero.
These counts share cases and donors and are descriptive, not independent
trials. The retained independent report contains the per-case endpoint bits,
effects and all four donor margins.

## Evidence and limits

Run `ba65abf8bbf7680ba28d2a2ebea67d45` has a complete controller terminal,
recorded owner exit 0 and a separately recorded controller exit 0, with matching
cooperative CPU-custody identities. Both unsafe markers were absent when
inspected after reaping. The outer launcher exit 0 was observed in the local
command session; it was **not originally saved as a separate exit file**. The
included retrospective observation records that distinction and is not an
independent process attestation. The roots and filesystem require cooperating
processes, and their identifiers are hashes rather than secret paths.

The [inventory](inventory.json) contains sizes and SHA-256 hashes for the
seven original JSON records, the compressed journal and the supplementary
checks. The originals were copied without changing their bytes. The gzip
decompresses to the original 9,490,808-byte journal with SHA-256
`f30b6ae11e4fbf5276d9f5a5dc3ae5206fd2a3c9f2fdf4252aae0a844fb6f7ee`;
the terminal SHA-256 is
`7637da956e49bbae690cfea45debe7fb89f6bb600ee85a75dbb28223553e48d5`.
The separate read-only preflight and marker/launcher observations are local
records outside the terminal's own artifact inventory. The included
standard-library [independent checker](independent-science-check.py) rebuilt
the case and family statistics and controls from the saved data, producing
[its report](evidence/independent-science-review.json). A separate custody and
source audit also checked the bound files and process records. These audits
establish internal consistency of retained evidence, not faithful execution.
Full mask logits were not retained: their stored digests cannot reconstruct
the original tensors. Selected `is` and `are` logit bits and patch vectors are
retained in the journal.

To replay the model-free check, first decompress `evidence/events.jsonl.gz` to
a temporary `events.jsonl` file. From this directory, pass that file and a copy
of the manifest from the frozen commit to the checker:

```text
python independent-science-check.py path/to/frozen-manifest.json path/to/events.jsonl evidence/result.json
```

The checker writes the reconstructed report to standard output and verifies
the journal digest. It does not load or run the checkpoint.
