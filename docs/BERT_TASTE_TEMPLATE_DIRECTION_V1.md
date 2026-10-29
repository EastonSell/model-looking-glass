# Crossed-template and direction-specific taste-context follow-up v1

## Question and scope

The preceding public matched-dose study's three eligible pairs all used
`after adding`, while all three `with some` pairs were ineligible. This
follow-up crosses six new noun stems with **both** templates to compare the
same stems under both contexts. Its primary analysis uses every completed
case, without filtering on clean-answer eligibility. It also compares the
effect of moving an ingredient activation toward and away from its donor with
the effect of one equal-dose orthogonal direction. Block output 3 was chosen
after seeing the earlier study and is therefore a motivated exploratory
choice. The two templates, six stems, one model and one artificial orthogonal
direction are not a representative population or evidence of a universal
taste representation.

Use the offline CPU float32 DistilBERT revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, deterministic algorithms, one
intra-op and one inter-op thread, and fixed seed `20261031`. Unknown previous
use and pretraining exposure are not certified. These are new frames held out
from the previous taste experiments, not secret cases or a blind confirmation;
the cases will be public before any model forward.

## New frames and selection

After this protocol and the case-selection source are public, rank this fixed
candidate pool by the bytewise SHA-256 digest of the ASCII bytes
`bert-taste-template-direction-v1`, followed by one zero byte and the UTF-8
stem:

`apple, berries, bread, cereal, chocolate, cream, custard, dessert, fruit,
jam, jelly, juice, lemonade, oatmeal, orange, porridge, rice, salad, soda,
syrup, toast, wine, yogurt`.

For each stem in digest order, consider exactly four rows: `The {stem} tasted
[MASK] after adding sugar.`, the same with `lemon`, and the two corresponding
`The {stem} tasted [MASK] with some {ingredient}.` rows. Accept the first six
stems for which the pinned tokenizer gives length 10, an all-one mask, exactly
one `[MASK]` at position 4, `the` at position 1 and exactly one sugar/lemon token difference
at position 7 in each pair. All four rows must share the same stem token at
position 2; CLS/SEP, `tasted`, punctuation and template-token IDs must match
the fixed pinned coordinates in the selection source. All four rows must be
distinct and have no exact or
either-way complete-row-prefix relation with an accepted row or with any of
the four cited public inventories below. Selection uses no model output.
If fewer than six stems pass, stop without inventing substitutes. Record the
pool, ordering, acceptance/rejection reasons and exact IDs/masks, then publish
the 24 selected rows and their inventory before the first pretrained forward.
The selection tool requires a clean checkout at the supplied full commit and
compares the selector, protocol, inventories and matched reservation with
their Git blobs; Git, interpreter and installed dependencies remain trusted.
Those checks bind bytes and do not themselves prove public chronology. No
result-driven selection, threshold change or redraw is allowed.

The cited inventory paths and SHA-256 values are:

* `experiments/bert_agreement_exposures_v6.json`: `0d0b37539c30c8ff25ea5b1cfa95ecd4587f048434348a1c91ec010418553afb`.
* `experiments/bert_matched_public_cases_v1.json`: `5b708600dba78db9e4ee676650bc5e1f6b3269d52179a8e45d5d45fee55219bb`.
* `experiments/bert_taste_public_exposures_v2.json`: `a766d20778191280122a6b498b9bcb34907bfac761152ae7c3cf4df0fb9c5aa1`.
* `experiments/bert_taste_matched_dose_public_exposures_v2.json`: `9b08e38927187c6f43d8659603e27e0f5397034e01143e665ab01a139a64e845`.

The last successor's rows and model identity are bound through its reservation
`experiments/bert_taste_matched_dose_public_cases_v1.json`, SHA-256
`983be330b6691ce23986f26e50a345f77ca5cf46dd28a660a3a012a2cf598a7c`.

## Doses and interventions

For each stem, capture both ingredient and `the` block-3 output vectors for
all four clean rows. For each template compute the sugar–lemon ingredient and
`the` distances in binary64 as `sqrt(fsum((a-b)**2 ...))`. The single radius
for that stem is the minimum of **all four** distances, so the two templates,
two positions and both directions are compared at the same requested L2 dose.
Distances and radius must be finite and at least `1e-5`; otherwise preserve
the partial evidence and stop without redraw. The same radius is applied to
each recipient independently; the actual post-float32 displacement must be
nonzero and within `1e-6 + 1e-4 * radius`. Record both requested and actual
norms, absolute/relative errors and ratios to each recipient norm (null with
reason `zero` if the latter is zero). This tolerance permits about 10% relative
error at the floor; the achieved doses are approximate, not exactly equal.

For each template, let `d` be the decoded sugar-minus-lemon ingredient
vector. Construct one deterministic 768-coordinate ±1 vector using successive
SHA-256 digests: seed is the SHA-256 of the ASCII bytes
`bert-taste-orthogonal-v1`, one zero byte, the UTF-8 stem, one zero byte, and
the ASCII template ID (`after` or `with`); hash
`seed + counter.to_bytes(4, "big")` for counters 0, 1, …, read each digest
most-significant bit first, and map 0 to −1 and 1 to +1 until 768 signs exist.
Compute `beta = math.fsum(v_i * d_i) / math.fsum(d_i ** 2)` in coordinate
order; compute each projected coordinate as `math.fsum((v_i, -beta * d_i))`,
its norm as `math.sqrt(math.fsum(projected_i ** 2))`, and `u_i` by separately
dividing the projected coordinate by that norm. If any input/projection is
nonfinite or the projected
norm is below `1e-8`, record the failure and stop; do not sample another vector.
The same `u` and `-u` are used for both recipients in the template.

For each row, in this order, run: full ingredient donor replacement; ingredient
self-patch; `the` self-patch; a radius-sized ingredient step **toward** the
other ingredient row; its equal-size opposite step; `+u`; `-u`; a radius-sized
`the` step toward the other row; and its opposite. Self-patches use their own
invocation's pre-action capture. Construct each non-self source from decoded
binary64 values with separately evaluated subtraction and multiplication,
`unit_i = (donor_i - recipient_i) / distance`,
`step_i = (sign * radius) * unit_i`, and `math.fsum((recipient_i, step_i))`,
then perform one integer-defined nearest-even binary32 rounding. For the
orthogonal action, substitute the corresponding `u_i` for `unit_i`. The CPU
runner must set and verify denormal preservation and probe it around capture,
source construction, intervention and scoring. Record raw
recipient, donor, source and observed before/after bits, achieved norms and
cosines with the appropriate recipient-to-donor clean direction. For each
recipient, `cos(toward, donor_direction) >= 0.99`,
`cos(away, donor_direction) <= -0.99`, and each orthogonal cosine must have
absolute value at most 0.01, after rounding;
otherwise mark the attempt incomplete. Both negative and orthogonal points can
be off-manifold. One orthogonal vector is not a generic null, and equal norm
does not equalize direction, naturalness or downstream sensitivity.

Before every action, its actual same-invocation site vector must match the
corresponding clean capture bit-for-bit, and its submitted and observed
post-action selected bits must match. The two clean captures per row, both
self-patch mask logits, the fresh-clean mask logits and the final-block donor
mask logits must equal the row's original clean mask logits bit-for-bit.
Any mismatch makes the attempt incomplete. Whole-mask equality is an owner
observation when full patched logits are not retained.

For each stem, eight clean captures precede the 36 intervention forwards;
four fresh clean checks follow. Separately capture the four final-block
ingredient vectors and perform four full donor replacements there as an
architectural no-later-attention control. This is 56 forwards per stem and
336 overall. Record every start/completion and all failures; no automatic
retry or replacement frame is allowed.

## Prespecified readout

For each forward, the score is the `[MASK]` logit of `sweet` minus `sour`;
effect is its patched score minus the recipient's clean score. Report full
clean-vocabulary logits, target probabilities and top-five tokens, with ties
ordered by descending logit then ascending token ID. Record target-logit bits
and summaries for every action, together with actual model-entry IDs and masks.
The prior clean eligibility rule is retained as a descriptive label only: the
sugar margin must be positive, lemon margin negative, and either target token
must appear among each row's top five. It never gates forwards or primary means.

Sugar→lemon means sugar donor into lemon recipient and uses direction sign +1;
lemon→sugar means lemon donor into sugar recipient and uses −1.
For each stem/template/direction compute the signed matched-position contrast
`sign * (ingredient_toward_effect - the_toward_effect)`. For each stem and
template, average the two directional contrasts with `math.fsum`; report both
templates' means across all six stems, each stem's `after adding` minus `with
some` difference, and the mean paired template difference. For each stem,
the direction main contrast is half the sum of `(A_sugar→lemon -
A_lemon→sugar)` across the two templates. The template-by-direction
interaction is the `after` directional difference minus the `with` directional
difference. Report each stem and the six-stem `math.fsum` means of both.
Also report every
underlying case and direction, including negative/tied effects and descriptive
eligibility. No p-value, threshold-driven victory claim or population
generalization is specified.

For each recipient also report `(effect(+)-effect(-))/2` and
`(effect(+)+effect(-))/2` separately for the donor-aligned, orthogonal and
contextual-`the` steps. Compare the donor-aligned odd component in the donor's
score direction with the *absolute* orthogonal odd component: the specified
direction-specific contrast is `sign * odd_ingredient - abs(odd_orth)`. For
each stem/template average that contrast over the two directions, then report
the six-stem means for both templates and their paired difference. Retain the
raw signed terms and even components so nonlinear or asymmetric responses
remain visible. Full-donor effects, both self controls, baseline logits/probabilities,
achieved norms/cosines and the final-block zero check are separate references.

An independently observed process exit, retained partial records on failure,
source/case identities and an independent saved-record numerical review are
required before reporting completion. Full untouched-output coordinates and
patched full-vocabulary summaries may be owner-observed when only hashes and
selected bits are saved. Recorded hashes and process observations do not prove
execution chronology, unmodified dependencies or whole-process-tree cleanup.
A GPU run would require a separate reviewed bounded owner, fresh device
admission and an explicitly recorded runtime; none is assumed here.
