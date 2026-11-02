# DistilBERT negation-cue transfer: CPU development study

The fixed directional-transfer criterion passed on six public evaluation
contexts: the mean effect was **+0.254324 logit**, with **6/6** context scores
positive. The predeclared rule required a mean strictly greater than 0.02 and
at least four positive scores. This is evidence of a directional effect of
these equal-norm state replacements on the model's `any` versus `some` masked
prediction; it is not a general test of negation or grammaticality.

## Design and outcome

The study used the cached `distilbert-base-uncased` checkpoint on CPU. Each
case had `not`, `never`, `just`, and `also` variants of the fixed template
`The {role} has {cue} {verb} [MASK] of the {object}.` Two cases were designated
development and six evaluation before any study forward. All are public
convenience cases: the evaluation rows were held out from this experimental
selection, not hidden or known to be absent from pretraining or other prior
exposure.

Let `L = logit(any) - logit(some)` and let a transfer's effect be its patched
target `L` minus that target's clean `L`. For each of `not` and `never`, the
directional score is half the difference between the cue-to-`just` and
`just`-to-cue effects at the block-2 cue position. A case score averages the
two cue scores. The 12 clean evaluation gaps `L(not/never) - L(just)` were all
positive, ranging from +5.07448 to +9.92631 logits.

| Case | Split and context | Directional score | Clean gap: not / never | Clean top token: not / never |
| --- | --- | ---: | ---: | --- |
| 01 | Development: doctor, examined, patients | +0.18051 | +7.66042 / +8.21720 | any / any |
| 02 | Development: chef, washed, dishes | +0.35005 | +7.12897 / +7.97540 | any / any |
| 03 | Evaluation: pilot, visited, airports | +0.28417 | +8.18949 / +8.74043 | any / any |
| 04 | Evaluation: teacher, read, books | +0.29361 | +9.22631 / +9.67873 | any / any |
| 05 | Evaluation: farmer, planted, flowers | +0.29683 | +9.81512 / +9.92631 | any / any |
| 06 | Evaluation: driver, inspected, vehicles | +0.19050 | +6.04795 / +6.32443 | any / any |
| 07 | Evaluation: scientist, measured, samples | +0.13189 | +5.07448 / +5.48136 | quality / quality |
| 08 | Evaluation: artist, painted, portraits | +0.32895 | +8.45909 / +8.95855 | any / any |

The full-vocabulary distinction matters: for case 07, `quality` was the top
masked prediction and `any` ranked third with `not` and second with `never`.
The saved result records the full-vocabulary top five and `any`/`some`
probabilities for each clean cue; the separately retained journal contains every full mask-logit
vector. A two-token margin is not a claim that either word is the model's
overall top prediction.

## Active comparisons and delivery

On the six evaluation cases, mean block-2 cue-position effects for a `just`
recipient were +0.433841 for `not` donors, +0.385866 for `never` donors and
−0.052900 for `also` donors. In all 12 negative-cue-to-`just` comparisons,
the positive negative-cue effect exceeded the absolute `also`-to-`just`
effect for the same recipient and dose. In the reverse direction, however,
the absolute `also`-to-negative effect was at least the absolute
`just`-to-negative effect in **11/12** comparisons. These are descriptive
active-control comparisons, not additional predeclared success criteria.
The reverse effect is not unique to the `just` donor. Because `also` and
`just` are both affirmative donors here, this comparison does not isolate
polarity-class information or refute a polarity-class effect.

The same bidirectional score averaged −0.015173 at the contextual object
position and +0.087488 at the mask position after block 2; these are active
transfer comparators, not expected nulls. It was zero at the final-block cue
position, where the positionwise output head has no subsequent cross-token
attention. The independent full-vector replay checked full-logit equality for
all self patches, final-cue transfers and final clean repeats, along with exact
patch landing and unchanged other token positions.

Each context's nonzero replacements used its common minimum of 20 specified
donor distances across sites. Across all eight contexts the requested radii
were 1.00714–1.83496, requested fractions of the natural donor displacement
were 0.05620–1.0, and the largest realized relative norm error among 256
transfers was approximately 6.90×10⁻⁸, below the fixed 0.001 tolerance. The
equal input norms do not ensure equal functional effects; a minimum based on
all sites can also make some patches small relative to their natural donor
displacements.

## Evidence and limits

According to the operator's observed launch chronology, the
[source and protocol were published](https://github.com/EastonSell/model-looking-glass/pull/479)
before the study forward. The recorded source revision was
`c72d120e4c22a56a9eda051893430631fb963c70`, with the same tree as the
independently reviewed local revision. The direct CPU invocation was observed
to exit 0; it reported 448 completed forwards and 456 records. The checkpoint
assets, selected project files, model state including nonpersistent buffers,
runtime and ordered records are bound by the saved hashes. The no-forward
preflight used the byte-identical earlier local source tree and blocked the
model forward, selected network calls and CUDA initialization. The actual run
used a clean detached checkout and reported no CUDA initialization.

Two separately written standard-library readers replayed the closed records.
`audit_full.py` checks the full saved vectors, site witnesses, unchanged token
bits, exact dose calculations, full-logit controls and fixed decision;
`audit_scores.py` separately reconstructs action order, target-logit effects
and the decision. Their output does not independently attest that the claimed
execution occurred, that local source or dependency bytes stayed unchanged
throughout execution, or that the inputs were hidden from pretraining. The
interpreter and installed dependencies were trusted rather than independently
content-attested.

The complete events.jsonl.gz journal is 85,368,356 bytes with SHA-256
a67b5ea0ab7d4f7078bab57ad315f9bcf90302be674e8723c05c0785a011ccd1.
It is omitted from this compact repository copy and is retained in a local
user-facing evidence bundle. No public journal download is provided by this
repository copy. The audit reports below are summaries of a replay
against that journal; a reader without the journal cannot reproduce the
full-vector checks from this compact copy alone.

After obtaining the original journal, place it beside these files and run
the full saved-data audit with Python's standard library:

```text
python -B audit_full.py manifest.json events.jsonl.gz result.json --expected-source c72d120e4c22a56a9eda051893430631fb963c70
python -B audit_scores.py manifest.json events.jsonl.gz result.json
```

The checkpoint and original journal are not included in this compact copy.
`inventory.json` lists the included files and SHA-256 hashes. Local chronology and recorded hashes describe the
retained evidence, not a general execution attestation.

This sentence family also has lexical and pragmatic confounds: `never` changes
temporal meaning, `just` changes focus or time, `also` has an additive
presupposition, and `any` can have free-choice uses. The results are limited
to these eight prompts, this checkpoint and this intervention. They do not
establish a unique location, a population-level effect, broad grammatical
knowledge, or the absence of an effect at other doses. Prior work motivating
these limitations includes [Warstadt et al. (2019)](https://aclanthology.org/D19-1286/),
[Kletz et al. (2023)](https://aclanthology.org/2023.nodalida-1.54/) and
[Zhang and Nanda (2024)](https://arxiv.org/abs/2309.16042).
