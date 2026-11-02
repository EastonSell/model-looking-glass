# Held-out cue-frame transfer: prospective CPU study

## Question and fixed cases

Does the prior same-recipient cue-class contrast persist when the cue and
masked item both occur in two clause configurations absent from the earlier
cue-class development and evaluation templates? The [earlier public study](https://github.com/EastonSell/model-looking-glass/blob/a13a2eab68d02cbf93f43af9ad01559364d129fd/results/bert-cue-class-cpu-v1-20261029/README.md)
passed its strict rule in 7/9 evaluation cases, but used one development
example from each evaluation template and contrasted `never`/`rarely` with
`always`/`often`. This follow-up is selected after that outcome was known.
It is a new prospective test of a fixed model-score contrast, not a hidden
sample, a population estimate or a test of general negation understanding.

Before any follow-up pretrained-model forward, fix exactly six public
evaluation contexts, three paired role/verb/object triples crossed with two
new templates:

* Subject relative: `The {role} who has {cue} {verb} [MASK] of the {object} has signed the form.`
* Causal adjunct: `The clerk is writing a note because the {role} has {cue} {verb} [MASK] of the {object}.`

The fixed triples are `inspector/examined/packages`, `assistant/sorted/reports`
and `worker/cleaned/tools`. There are no development examples of either new
frame, no tokenizer-driven lexical or outcome-driven case replacement, and no model forward
for an exploratory variant of a new frame. A tokenizer-only probe and
independent scientific review rejected an earlier proposal in which the cue
and mask remained in the prior matrix-clause configuration; the candidate
is recorded in the manifest. Verify the 24 fixed texts and token IDs with the
pinned tokenizer before model loading. The triples are paired across frames,
not six independent lexical samples. Individual words, including `examined`,
occur in prior public cases; the complete new texts and frames are public and
not known to be absent from pretraining.

The operational negative/minimizing cues are `not` and `hardly`; the
affirmative cues are `really` and `certainly`. The common `has` holds the
auxiliary and aspect fixed, and the same triples constrain cross-frame lexical
differences. Replacing the earlier frequency-adverb set does not establish a
reduction in lexical-frequency confounding. These choices do not
make the cues semantically matched: `hardly` is a scalar minimizer, not the
same logical operator as `not`, and `really` and `certainly` differ in
intensity, certainty and distribution. `any` has free-choice uses, and causal
and relative clauses can introduce pragmatic readings. The two frames also
differ in prefix, position, length and discourse meaning, so differences cannot
be attributed to syntax alone. They are held out from the earlier experimental
templates, not from pretraining. No model-score threshold is a human
acceptability judgment.

NPI behavior depends on the licensor and evaluation method (see
[Warstadt et al., 2019](https://aclanthology.org/D19-1286/) and
[DeCarlo et al., 2023](https://aclanthology.org/2023.blackboxnlp-1.25/)).
The interaction of `because` and polarity is context-dependent; simple
unnegated `because` alone does not license weak NPIs in the examples discussed
by [Hsieh, 2012](https://linguistics.uconn.edu/wp-content/uploads/sites/1842/2020/05/HsiehI_2012_Polarity-in-Conditionals-and-Conditional-like-Constructions.pdf),
but that observation does not settle the interpretation of these sentences.

## CPU runtime and matched delivery

Use the pinned `distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, CPU float32, eager attention,
evaluation mode, one intra-op and one inter-op thread, deterministic algorithms,
no autocast or network, and seed `2026102903`. Set `CUDA_VISIBLE_DEVICES=-1`
before importing Torch; never initialize or use CUDA. Use the earlier
source-bound owner's pinned assets, versions, nonpersistent-buffer-inclusive
model-state digest, and before/after subnormal no-flush checks. Record selected
source and manifest hashes, package versions, CPU environment and state digests.

Capture the cue, object and mask positions after block 2 and the cue position
after final block 5. For each context compute all 24 L2 distances among the
six unordered cue pairs at the four sites. With coordinate-order binary64
arithmetic, `distance(x,y) = sqrt(fsum((float(y_i)-float(x_i)) ** 2))`; set
the common radius `r` to their minimum. A finite `r < 0.001` makes that context
dose-ineligible: retain its four clean forwards, all distances, named
ineligibility and four final clean repeats, then continue. Do not amplify or
replace the case. Nonfinite values, incomplete calls or failed controls make
the attempt incomplete rather than negative.

For every nonself transfer, use `alpha = r / donor_distance` in binary64,
require `0 < alpha <= 1`, calculate separate binary64 subtraction,
multiplication and addition, then round once to nearest-even float32. Require
the same-forward pre-action target to bit-match its clean capture, the
selected landed vector to bit-match the submitted vector, all other token
coordinates to remain unchanged, and the realized displacement's relative
error to be at most 0.001. Save full before/after states, selected vectors,
fraction, requested and realized norms, and the complete 30,522-logit mask
vector. Hidden states must have width 768. Preserve any bounded returned
witness on a subsequent failure when feasible; report unavailable or oversized
components as omitted.

The fixed order is the manifest's six cases; cue order `not`, `hardly`,
`really`, `certainly`; site order `cue2`, `object2`, `mask2`, `cue5`; at each
site four self patches and all 12 distinct ordered transfers in target-then-
donor cue order. Four clean forwards precede the patches and four clean
repeats follow them. Thus a dose-eligible context uses 72 forwards and the
maximum is 432. Clean and final-repeat forwards use passive capture hooks,
not an independently hook-free baseline. Self and final-repeat full mask
logits must equal the clean target; final-block cue transfers must also leave
the complete mask-logit vector unchanged. Contextual object and mask states
are active comparators, not assumed nulls.

## Screen and specificity decision

Let `L = logit(any) - logit(some)` and define a transfer effect as patched
target `L` minus clean target `L`. A context passes the clean baseline screen
only if all four gaps `L(n) - L(a)` are strictly greater than 0.5 for both
negative/minimizing cues `n` and both affirmative cues `a`. A tie fails the
screen. Retain every gap and all clean logits/probabilities. Baseline-ineligible
contexts still receive every planned patch if dose-eligible; they are not
discarded, replaced or counted as primary passes.

At the block-2 cue site, for each target let the sign be +1 for an affirmative
recipient and -1 for a negative/minimizing recipient. Take the minimum of the
two signed opposite-class donor effects, and subtract the absolute effect of
the other same-class donor into that same target at the same requested radius.
Each of the four target margins must be strictly greater than 0.05 for an
eligible context to pass. This rule fails a target when a same-class donor is
comparable to the weaker predicted opposite-class donor. Retain every raw
donor effect, signed opposite effect, same-class effect, target margin and
failure, including those of ineligible cases.

The primary rule is evaluable only when at least two of three contexts in
**each** frame pass both baseline and dose screens. If any frame has fewer
than two eligible, report `unevaluable`, not a pass or failure. Otherwise,
the primary rule passes only if at least two contexts in each frame pass all
four target checks; it fails otherwise. The passing contexts need not be the
same two triples across frames. Report both frames, all six case statuses and
their paired per-triple changes; paired changes are descriptive, not an
additional criterion. Ties at either threshold fail. Do not retune thresholds,
alter cases or omit negative results after outputs.

Report all four-site effects and comparator margins, the clean full-vocabulary
top five and `any`/`some` probabilities, all baseline gaps, radius and dose
fractions, and both eligible and ineligible counts. Equal input norms do not
match functional impact. The sample is conditional on the clean screen and is
too small and correlated for a population or statistical-significance claim.

## Evidence and exposure

Freeze and independently review the exact cases, protocol and executable
source before the first follow-up forward. Save completed bounded witnesses
and mask logits in ordered, sync-flushed gzip JSONL; retain incomplete attempts
and their failure phase. Independently replay the closed records, including
full-logit controls, dose, landing, unchanged tokens, screen and fixed decision,
without another model forward. The direct CPU exit is an operator observation;
source/runtime hashes are saved-byte checks under a trusted, quiescent source
and dependency environment, not execution attestation.

The [public ID/mask comparison](../experiments/bert_cue_heldout_frames_public_exposure_check_v1.json)
checks the 24 input rows against eight selected public BERT inventories,
including both prior cue studies, before forwarding. Such a comparison is not
a private-split, complete-history,
distributional-independence or pretraining-exposure attestation.
