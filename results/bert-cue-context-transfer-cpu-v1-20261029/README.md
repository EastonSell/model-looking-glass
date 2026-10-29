# Same-cue state transfer across contexts

At the block-2 cue site, a same-cue state from another context moved the
`any`−`some` masked-logit difference by at least half as much on average as a
different same-class cue from that **same donor context** in both fixed
evaluation pairs. In fact its mean absolute movement was larger in both pairs.
This met the prospective magnitude rule; it did not establish directional
agreement, semantic invariance, or a general negation mechanism.

| Fixed pair | Mean absolute same-cue shift A | Mean absolute same-class shift B | H = A − B/2 | Outcome |
| --- | ---: | ---: | ---: | --- |
| Discovery: nurse/charts ↔ ranger/birds | 0.009645 | 0.019170 | 0.000060 | at least half |
| Evaluation: librarian/books ↔ carpenter/shelves | 0.037221 | 0.022397 | 0.026022 | at least half |
| Evaluation: editor/articles ↔ musician/songs | 0.022663 | 0.021634 | 0.011846 | at least half |
| Both evaluation pairs pooled | 0.029942 | 0.022016 | 0.018934 | at least half |

The rule compares the average of eight matched recipient/direction
magnitudes per pair and accepts `H >= 0` only when the pair's baseline and
dose screens pass and `B >= 0.01`. Both evaluation pairs were eligible and
informative; the discovery pair is separate and its H is very close to zero.
Pooling averages all 16 recipient magnitudes; larger effects contribute more
to the sum, so the pair rows matter.
The full-precision per-recipient signed effects are in [recipients.csv](recipients.csv).
In 5 of 16 evaluation recipients, the same-cue and same-class effects had
opposite signs. For example, on the carpenter/`rarely` recipient their shifts
were −0.043892 and +0.031918, respectively. The magnitude rule does not test
whether they move a score in the same direction.

The broader output comparison gives a different picture. At the cue site,
average full-vocabulary total variation from the clean recipient was **0.005262
versus 0.007162** in evaluation pair 1 and **0.005230 versus 0.006049** in
pair 2, for same-cue versus same-class donors respectively. These descriptive
TV comparisons do not have a separate success threshold. After removing
`any` and `some` and renormalizing the remaining vocabulary, the corresponding
averages were 0.004795 versus 0.007002 and 0.005909 versus 0.006680. The
24 clean distributions assigned 0.01634–0.80205 probability mass to
`any` plus `some`; the remaining mass used for conditional TV was therefore
0.19795–0.98366. See [tv_summary.csv](tv_summary.csv) and [effects.csv](effects.csv)
for all sites, donors, target-pair masses and individual values.

This is one shared sentence template, two public evaluation pairs and one
public discovery pair. The cases were fixed before this follow-up's outputs
but were visible during design and are not known absent from pretraining.
The baseline screen required each negative/low-frequency cue to exceed each
affirmative/high-frequency cue by more than 0.5 in the clean two-token
contrast in both contexts. All three pairs passed; this is a model-score
condition, not a human acceptability judgment. For the editor/`rarely` input,
`versions` was the top masked token, illustrating why this contrast is not
the model's complete prediction. The cues differ in frequency and temporal
meaning, and context wording changes role, verb and object together. A
same-cue patch is itself an active intervention. Equal input norms do not
equalize direction or functional effect.

Each pair used one fixed-rule common requested norm q=0.5, with minimum
natural donor distances 1.151180, 0.986813 and 1.367893 in table order. No
requested transfer amplified a donor displacement. Achieved norms ranged
0.499999946–0.500000056, with maximum relative error below 1.127e−7. The
run saved 816 forwards and 822 events: 24 clean, 96 self, 672 transfer,
24 repeat, three baseline and three radius records. Clean/repeat runs used
passive capture hooks. Self and repeat full mask logits matched clean, and
all final-block cue patches left the full mask logits unchanged. Object and
mask transfers were active: across all pairs their `any`−`some` shifts ranged
−0.04744 to +0.01893 and −0.13417 to +0.12789 respectively. All opposite-class
and within-context comparisons remain in [effects.csv](effects.csv).

The operator directly waited for the CPU process and observed exit 0. The
saved result records equal before/after model-state hashes and CUDA not
initialized at the final check. An independent full-witness replay checked
geometry, landings, untouched positions and full-logit controls; a separate
score replay checked full-vocabulary TV, baseline summaries and the outcome.
Their reports are [full-witness-audit.json](full-witness-audit.json) and
[score-tv-audit.json](score-tv-audit.json). These are checks of saved records
and reported identities under trusted local execution, not execution or
pretraining attestations.

The recorded execution source is
[`40752bde00f11b69832066c2d16ce52c49f96810`](https://github.com/EastonSell/model-looking-glass/tree/40752bde00f11b69832066c2d16ce52c49f96810).
The later [source merge](https://github.com/EastonSell/model-looking-glass/commit/91d4a3913dacd05436e3ce5248f96aaf2b7aefd9)
retained a different version of the shared intervention helper, so use the
recorded source for exact reproduction. The full-witness reviewer reports
that the reader was fixed and publicly read back before examining the study
outcome. The score/TV reviewer reports that its reader was fixed before outcome
inspection and published after process exit. The independent public-input
checks are scoped to their named snapshots, including one explicitly identified
planned color manifest; they do not cover unknown or private exposure.

The complete 161,286,491-byte compressed journal is retained locally on E:
scratch, SHA-256 `93ff93b3b7a563b16a14cb24c81e52e188cca4c98ea4abefbb0fd509f797fbd9`.
It is not included in this repository copy and has no public download location.
The available journal hash is a commitment, not the journal itself. The
37,131-byte operator-derived [score subset](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/bert-cue-context-transfer-cpu-v1-20261029/bert-cue-context-score-tv-subset-40752bde.jsonl.gz)
retains all selected float32 `any`/`some` pairs and copies the independently
audited full/excluded TV and target-mass values from the result. Its reader
replays selected scores, the fixed eligibility and decision, and the arithmetic
of those reported TV summaries. Two logits cannot reconstruct full-vocabulary
TV, and the subset cannot authenticate its derivation, full logits, activation
geometry, dose, landing or execution. The full journal and independent audits
support those saved-record checks locally.

From this compact package, a model-free subset replay is available:

```text
python -I -B review_bert_cue_context_subset.py manifest.json bert-cue-context-score-tv-subset-40752bde.jsonl.gz result.json --expected-manifest-sha256 4fb31c4b49c24e78f6a4b0b59d928ebb6204a55ecdc2c9c4cd55528700d592a7 --expected-subset-sha256 1d5cb23b1a8156c2a621de9c351253eba76c21c6346039df6267fe47dee3297a --expected-result-sha256 4f3eca14fffea1df6d8504d79017c2980ba03f8f31fd3fd7b7d1812f3697cbc8 --expected-original-journal-sha256 93ff93b3b7a563b16a14cb24c81e52e188cca4c98ea4abefbb0fd509f797fbd9 --expected-source 40752bde00f11b69832066c2d16ce52c49f96810 --expected-started-sha256 218914aded4d864304d444759aad20985d4e68072bbba05922ef43965e288116
```

With the full journal available as `JOURNAL`, the standard-library full-witness
replay can be run from this directory:

```text
python -I -B audit-full.py manifest.json JOURNAL result.json --started started.json --expected-started-sha256 218914aded4d864304d444759aad20985d4e68072bbba05922ef43965e288116 --expected-result-sha256 4f3eca14fffea1df6d8504d79017c2980ba03f8f31fd3fd7b7d1812f3697cbc8
```

The separate full-vocabulary score replay uses the same inputs:

```text
python -I -B review_bert_cue_context_scores.py manifest.json JOURNAL result.json --started started.json --expected-source 40752bde00f11b69832066c2d16ce52c49f96810 --expected-started-sha256 218914aded4d864304d444759aad20985d4e68072bbba05922ef43965e288116 --expected-result-sha256 4f3eca14fffea1df6d8504d79017c2980ba03f8f31fd3fd7b7d1812f3697cbc8
```

The five CSV tables can be regenerated without a model from the retained
manifest and result: `python -I -B make-tables.py manifest.json result.json TABLES`,
where `TABLES` is a new directory. The [protocol](protocol.md) defines the
fixed comparisons, rounding order, exclusions and candidate-selection history.
