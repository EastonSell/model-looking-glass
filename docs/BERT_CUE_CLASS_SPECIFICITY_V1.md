# Prospective cue-class transfer across three frames

## Question and context

Can equal-norm DistilBERT cue-state replacements move a fixed masked
`any`-versus-`some` preference in the expected cue-class direction more than
a same-recipient, same-class lexical replacement? The earlier
[cue-transfer study](https://github.com/EastonSell/model-looking-glass/blob/a87d0916248f9fa64bf76134843afd5f06fddec2/results/bert-negation-cue-cpu-v1-20261029/README.md)
found a directional effect, but its `also`-to-negative and `just`-to-negative
comparisons both used affirmative donors. Their similarity shows the reverse
effect is not unique to `just`; it does not test a within-negative donor or
refute a polarity-class effect.

This follow-up uses within-class donors for **both** target classes. It is a
conditional, exploratory intervention on one model's masked-token score, not
a test of general negation understanding or sentence acceptability. Prior
NPI research has found that model assessments vary with method and licensing
context ([Warstadt et al., 2019](https://aclanthology.org/D19-1286/),
[DeCarlo et al., 2023](https://aclanthology.org/2023.blackboxnlp-1.25/));
clause-level lexical effects can also confound interpretation
([Kletz et al., 2023](https://aclanthology.org/2023.nodalida-1.54/)).

## Fixed public cases and chronology

Before any follow-up pretrained model forward, freeze
`experiments/bert_cue_class_public_cases_v1.json`: three development contexts
and nine evaluation contexts, three evaluation contexts per frame. Each case
has four variants, differing only at a single cue token:

* `main`: `The {role} {cue} {verb} [MASK] of the {object}.`
* `embedded`: `The manager said that the {role} {cue} {verb} [MASK] {object}.`
* `relative`: `The {role} who met the clerk {cue} {verb} [MASK] of the {object}.`

The operational negative/low-frequency class is `never`, `rarely`; the
high-frequency affirmative class is `always`, `often`. `rarely` is not
explicit sentential negation, and all four words vary in frequency and temporal
meaning. `any` also has free-choice uses. A successful class contrast could
reflect those factors, token identity, or contextual distribution rather than
abstract polarity. The embedded frame has a nonpartitive masked noun phrase;
the relative frame changes the subject phrase. These examples and their
evaluation labels are public convenience choices, not blinded cases or an
independent sample from pretraining. The contexts within each frame share
syntax, so their counts are descriptive rather than independent trials.
In the relative frame the cue and mask are in the main clause, after a
subject-modifying relative clause. The frames also differ in partitivity,
lexical choices, sequence length and token positions; differences between
frame summaries cannot be attributed to syntax alone.

The cases and criterion were designed after inspecting the earlier study's
results; a preliminary, not independently verified, full-vocabulary analysis of
that earlier study was also received before this protocol was frozen. No
follow-up model output has been used to choose the cases, screen, sites or
thresholds. A tokenizer-only check replaced `mailed` with single-token `sent`
before any follow-up forward; that rejected candidate is retained in the
manifest. Verify all 48 rows, positions, all-one masks and the single-token
`any`/`some` IDs with the pinned tokenizer before model loading.
An exact token-ID and attention-mask comparison with seven selected public
BERT exposure/case inventories is recorded in
`experiments/bert_cue_class_public_exposure_check_v1.json`. It found no exact
matches among those inventories; it does not check private splits, establish
distributional independence or certify complete history or pretraining.

## Runtime, sites, and matched displacement

Use the same cached `distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, CPU float32 eager attention,
evaluation mode, one intra-op and one inter-op thread, deterministic algorithms,
no autocast or network, and Torch seed `2026102902`. Set
`CUDA_VISIBLE_DEVICES=-1` before Torch import; never initialize or use CUDA.
Use the exact asset hashes and installed package versions recorded in the
earlier source-bound CPU owner; record the selected source and manifest hashes,
runtime, CPU environment and model-state digest before and after. The digest
includes the state dictionary and all named buffers, including nonpersistent
buffers. Disable float32 denormal flushing and verify a float32 subnormal
before and after the forwards.

Capture the cue, object and mask token after block 2 and the cue after final
block 5; token indices are fixed per frame in the manifest. Bidirectional
contextual object and mask states are active comparators, not assumed nulls.
At the final cue position, no later layer mixes positions, so a cue-only
patch must preserve the entire mask-logit vector exactly.

For each case compute all 24 distances for the six unordered cue pairs at
all four sites and set `r` to their minimum. For vectors `x` and `y`, distance
is `sqrt(fsum((float(y_i)-float(x_i)) ** 2))` in coordinate order. Require a
finite `r >= 0.001`; record every distance and every binding pair/site. For
each nonself replacement use binary64 `alpha = r / distance` with
`0 < alpha <= 1`, then separate binary64 subtraction, multiplication and
addition followed by one nearest-even float32 cast. Measure the realized
displacement from the same-forward pre-action target vector and require
relative norm error at most 0.001. Record the fraction, requested and realized
norm, full target/donor/submitted/landed vectors and whole-block before/after
witnesses. The same-forward target must bit-match its clean capture, the
landed selected vector must bit-match the submitted vector and every other
token coordinate must be unchanged.

If the radius is finite but below the floor, keep the four clean baselines,
all distances and a named `dose_ineligible` record; do not substitute or
amplify a dose. Run and retain the four final unpatched repeats for that case,
then continue to the next case. Such a case is ineligible for the primary
criterion. A nonfinite value, rounding/landing failure, failed control or
interrupted/partial model call makes the attempt incomplete rather than
becoming a negative result. Retain every bounded witness already returned and
the named failure where feasible.

For every dose-eligible case: run four clean variants; at each site run four
same-forward self patches and all 12 ordered transfers between distinct cues;
then run four unpatched repeats. The full schedule is 72 forwards per case,
864 for 12 dose-eligible cases. Self-patched and final repeated full mask
logits must equal the corresponding clean target bits; final-cue transfers
must also equal their clean target mask-logit bits. Save the full float32 mask
vocabulary vector for every completed forward, not only `any` and `some`.
The real owner requires 30,522 vocabulary logits and 768-dimensional captured
states at every forward; a bounded malformed result is retained as a named
failed witness, while unavailable or oversized components are labeled omitted.
Use cue order `never`, `rarely`, `always`, `often`, site order `cue2`,
`object2`, `mask2`, `cue5`, and for transfers iterate targets in cue order,
then the three distinct donors in cue order. Execute cases in manifest order.

## Fixed baseline screen and specificity decision

Let `L = logit(any) - logit(some)` and
`delta(donor -> target, site) = L(patched target) - L(clean target)`. The
baseline screen is `min[L(n)-L(a)] > 0.5` over both negative-class cues `n`
and both affirmative-class cues `a`; a tie or smaller gap is baseline
ineligible. Save all four gaps and every clean margin and probability. The
screen only conditions the primary evaluation; even a baseline-ineligible
case receives and retains the full intervention schedule whenever its dose is
eligible. Do not select replacement cases or alter thresholds after outputs.

At `cue2`, for each target `t` let `s_t` be +1 for an affirmative target and
−1 for a negative-class target. Define

`cross_min(t) = min_d [s_t * delta(d -> t, cue2)]`, where `d` ranges over
the two cues in the opposite class. Let `same_abs(t)` be the absolute effect
of the other cue in `t`'s own class, patched into the same target at the same
requested radius. The strict target specificity margin is
`cross_min(t) - same_abs(t)`; a target passes only if this is **strictly
greater than 0.05** logits. This requires *both* signed opposite-class donor
effects to exceed the magnitude of the matched same-class donor effect by
more than 0.05, so a comparable unrelated same-class effect fails that target.
A context passes only if all four targets pass and its baseline and dose are
eligible. Ties fail. Record every individual signed and raw effect, same-class
absolute effect, target margin, eligibility reason and context decision.

For the nine evaluation cases, the primary criterion is **evaluable** only
if at least two of the three cases in every frame are baseline- and
dose-eligible. When evaluable, it passes only if at least two cases in *each*
frame pass all four target checks; otherwise it fails. If any frame has fewer
than two eligible cases, report the criterion as unevaluable, not passed or
failed. Retain all nine evaluation rows, including baseline-ineligible,
dose-ineligible and negative outcomes, along with all three development rows.
The reported success rate is conditional on the baseline screen; it is not a
population estimate or p-value.

Report the same individual cross/within effects and scores at `object2`,
`mask2` and `cue5`; clean full-vocabulary top five and the `any` and `some`
probabilities; baseline gaps; per-case dose/fractions; and both eligible and
ineligible counts. Equal input norms do not equalize functional impact.
Frame, frequency and lexical confounds remain even when the criterion passes.

## Evidence

Freeze and independently review the protocol, cases and executable source
before the first follow-up model forward. Save each completed bounded witness,
full mask logits and controls in ordered gzip JSONL, sync-flushing each event;
retain incomplete attempts and their failure phase without promoting them.
The direct CPU process exit is an operator observation. Independently replay
the saved schedule, dose, all patch/unchanged-token witnesses, full-logit
controls, baseline screen and primary decision without another model forward.
Source/runtime hashes are saved-byte evidence under a trusted, quiescent
checkout and dependency environment, not an attestation of execution or prior
exposure.
