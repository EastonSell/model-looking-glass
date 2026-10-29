# Taste-context propagation and matched-disruption follow-up v1

## Question and scope

In the preceding public study, transferring the opposite ingredient's
contextual `the` vector moved the sweet–sour score opposite the donor direction
in all 24 reported cells. That observation does not establish whether the
response reflects ingredient information propagated into that position,
general sensitivity to a particular activation direction, or both.

This prospective public follow-up makes two complementary interventions. It
patches only the ingredient output of block 0 and captures the later outputs at
the ingredient, `the`, `tasted` and `[MASK]` positions in the *same forward*.
It also patches the ingredient, `the` and `tasted` outputs of block 3 and
compares the resulting full mask-logit vectors. Both stages use the same
per-stem dose and the matched controls below. The first tests whether later
positions respond to a localized early intervention; the
second compares downstream sensitivity at the three later positions. Neither
isolates a unique route or proves that a position represents a general taste
concept.

Use six new public noun stems, each crossed with both fixed templates and both
ingredients. The two templates and block 3 were motivated by prior results;
these are convenience cases, not independent samples or a blind confirmation.
No clean-answer eligibility filter is applied. Prior unknown use and model
pretraining exposure cannot be excluded. Use offline CPU float32
`distilbert/distilbert-base-uncased`, revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, fixed seed `20261102`, one
intra-op and one inter-op thread and deterministic algorithms. A GPU study
requires a separately reviewed device-aware owner and fresh admission; it is
not a prerequisite for the CPU study.

## Public case selection

After this protocol and selector are published, rank the following fixed pool
by SHA-256 bytes of ASCII `bert-taste-context-propagation-v1`, a zero byte and
the UTF-8 stem:

`apple, berries, bread, cereal, custard, fruit, jam, jelly, juice, lemonade,
oatmeal, orange, porridge, soda, toast, wine, yogurt`.

For each stem in digest order, construct exactly four public rows:
`The {stem} tasted [MASK] after adding sugar.`, the same with `lemon`, and
the corresponding two `The {stem} tasted [MASK] with some {ingredient}.`
rows. Use only the pinned local tokenizer, whose `tokenizer.json` SHA-256 is
`ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98`;
no model output participates in selection. Accept the first six stems whose
four rows have the exact fixed length-10 IDs and all-one masks specified by
the selector: CLS/SEP and punctuation, exactly one mask at position 4, `the`
at 1, one shared stem token at 2, `tasted` at 3, the declared template tokens
at 5–6, and sugar/lemon as the sole pairwise difference at position 7.

Each accepted row must be distinct and have no equality or either-way
complete-row-prefix relation with another accepted row or any of these pinned
public inventories. Validate each inventory's bytes, model/revision and
reservation binding as applicable:

* `experiments/bert_agreement_exposures_v6.json`: `0d0b37539c30c8ff25ea5b1cfa95ecd4587f048434348a1c91ec010418553afb`.
* `experiments/bert_matched_public_cases_v1.json`: `5b708600dba78db9e4ee676650bc5e1f6b3269d52179a8e45d5d45fee55219bb`.
* `experiments/bert_taste_public_exposures_v2.json`: `a766d20778191280122a6b498b9bcb34907bfac761152ae7c3cf4df0fb9c5aa1`.
* `experiments/bert_taste_matched_dose_public_exposures_v2.json`: `9b08e38927187c6f43d8659603e27e0f5397034e01143e665ab01a139a64e845`.
* `experiments/bert_taste_template_direction_public_exposures_v2.json`: `bf4d67ca16519a133f263d2da4b7a270bcd7a259a7cab4b8e8c85a04baa4f424`.

The last inventory's 24 rows must equal reservation
`experiments/bert_taste_template_direction_public_cases_v1.json`, SHA-256
`e59f5e1da0b3ce8c99ea9c231b4998e1c1e685d09c20bff4fdb734dc4910256f`.
The matched-dose successor is bound through its reservation, SHA-256
`983be330b6691ce23986f26e50a345f77ca5cf46dd28a660a3a012a2cf598a7c`.
Record the entire candidate ordering, rejection reasons, selected text, IDs and
masks and the selector/protocol/inventory source hashes. Publish the resulting
24-row reservation and an operator-observed selector exit before the first
pretrained forward. If fewer than six stems pass, stop; no substitution or
outcome-driven redraw is allowed. These checks cover only the named inventories.

## Intervention geometry and controls

For each stem, clean-capture all four rows at block-0 ingredient position 7
and block-3 positions ingredient 7, `the` 1, `tasted` 3 and `[MASK]` 4. At
each of the four intervention sites—early ingredient and the three block-3
sites—compute four clean Euclidean distances: sugar versus lemon within each
template, and `after` versus `with` for each ingredient. Use decoded float32
values in binary64 and `sqrt(math.fsum((a-b)**2 ...))` in coordinate order.
The common radius is the minimum of all 16 distances for that stem. If any
distance or radius is nonfinite or below `1e-5`, retain the partial attempt and
stop, without replacement. The same radius is used across both templates,
both recipients, both blocks, all sites and all actions for that stem.

At each recipient/site, define the lexical donor as the opposite ingredient
in the same template and the contextual donor as the same ingredient in the
other template. In order, apply: lexical toward and away; contextual toward
and away; two deterministic orthogonal directions, each with both signs; and
a same-invocation self-patch. The cross-template direction changes real
context while retaining the ingredient: it is a contextual comparator, not a
neutral null. Orthogonal points and all away steps may be off-manifold. Two
fixed orthogonal directions do not characterize all unrelated perturbations.

For each stem/template/site, make each orthogonal sign vector by SHA-256 of
ASCII `bert-taste-context-propagation-orth-v1`, a zero byte, UTF-8 stem, a
zero byte, ASCII template (`after` or `with`), a zero byte, ASCII site
(`early_ingredient`, `late_ingredient`, `late_the`, `late_tasted`), a zero
byte, and ASCII index `1` or `2`. Hash that 32-byte seed plus successive
unsigned 32-bit big-endian counters, read digest bits most-significant first
and map 0 to −1 and 1 to +1 until 768 signs exist. Project the first sign
vector away from the clean sugar-minus-lemon site direction and normalize.
Project the second away from that lexical direction and then away from the
first normalized orthogonal vector, and normalize. In the second projection,
divide its dot with the first vector by `math.fsum(u1_i**2)` rather than
assuming its rounded binary64 norm is exactly one. Each projection dot and
squared norm uses `math.fsum` in coordinate order, with `x**2` for each
squared coordinate; subtraction uses
`math.fsum((x, -coefficient*y))` per coordinate, and each norm is
`math.sqrt(math.fsum(x**2 ...))`. A nonfinite or below-`1e-8` projected norm
ends the attempt without a redraw. The same two directions are used for both
ingredient recipients in that template/site.

For a donor direction, compute `unit_i = (donor_i-recipient_i)/distance`;
for an orthogonal action use its normalized vector. Evaluate
`step_i = (sign*radius)*unit_i`, then `math.fsum((recipient_i, step_i))` and
round once to binary32, nearest-even, using the integer-defined converter.
Record raw recipient, donor, submitted source and actual pre/post bits. The
same-invocation pre-action bits must equal the corresponding clean capture;
the selected post-action bits must equal the source, and the full-output
untouched-coordinate witness must pass. Actual displacement must be nonzero
and within `1e-6 + 1e-4*radius`; record actual norm, error, recipient norm and
ratio (null with reason if the latter is zero). The requested match is
approximate: at the floor the absolute term allows about 10% relative error.
Post-rounding lexical/context toward and away displacements must have cosine
at least +0.99 and at most −0.99, respectively, with their own clean donor
directions. Both orthogonal signs must have absolute cosine at most 0.01 with
the lexical direction. For each recipient/site, all four pairs of achieved
displacements `(orth1 ±, orth2 ±)` must also have absolute mutual cosine at
most 0.01; record those four cosines and their cosines with the cross-template
direction. A failed geometry or witness makes the attempt
incomplete. Probe float32 and positive/negative binary64 subnormal preservation
before/after capture, construction, intervention and scoring, including each
whole-output analysis and aggregate calculation. Fail if the scoring thread
flushes these values; ordinary softmax underflow remains permitted.

## Fixed schedule and retained observations

For each stem, first run the four clean rows with block-0 ingredient and all
four block-3 captures in the same forward. Then, for each row and each of the
four intervention sites in the order above, run all nine actions in the stated
order. For every block-0 ingredient action, capture block-3 ingredient,
`the`, `tasted` and `[MASK]` vectors in that same forward after the earlier
patch. Then run four fresh clean checks, four final-block ingredient captures
and four non-self final-block ingredient donor patches. This is
`4 + 4*4*9 + 4 + 4 + 4 = 160` forwards per stem, 960 overall.

Retain the actual model-entry IDs/masks; each start and validated completion;
all clean and same-forward downstream selected float32 bits; the source and
landing bits, action metadata and geometry; and all baseline full-vocabulary
mask logits. Also retain the full 30,522 float32 mask logits of every non-self
matched intervention, so full-output analyses can be independently replayed.
Self patches, fresh clean checks and final-block donor patches must reproduce
the clean mask logits bit-for-bit. Every block-0 self-patch must also reproduce
all four later selected capture vectors, and every fresh-clean check must
reproduce all five clean site captures bit-for-bit. A later check or write failure after the
adapter returns should retain a bounded invalid-return event when possible,
but an adapter exception, interruption at its return boundary or failed write
may leave only a start. Preserve such attempts as incomplete; do not retry
silently. Whole-output untouched-coordinate equality remains an in-run witness
unless the whole tensors are retained.

## Prespecified readout

The target score is `[MASK]` sweet logit minus sour logit. For each recipient,
site and perturbation axis, report the raw `+` and `−` target effects relative
to that recipient's clean score, their odd component `(effect+ - effect−)/2`
and even component `(effect+ + effect−)/2`. For lexical orientation use sign
`+1` when sugar is the donor and `−1` when lemon is the donor. The primary
taste-specific contrast is the signed lexical odd component minus the mean of
the *absolute* odd components for the two orthogonal controls at the same
site and dose. Also report the signed lexical odd component minus the
absolute cross-template odd component, and every underlying signed component.
These are descriptive comparisons; the contextual comparator is not assumed
to be inert and equal input norm does not equalize downstream sensitivity.

For every non-self matched action compute total-variation distance from its
recipient's clean full-vocabulary softmax. Compute probabilities in binary64
using a max-subtracted exponential and coordinate-order `math.fsum`, and TV
as `math.fsum(abs(p_i-q_i))/2`. Report this distance for the `+` and `−`
distributions separately for each axis; TV is not computed on the odd logit
vector.
For each ± pair, let `V` be the coordinate-wise half-difference of the two
mask-logit vectors, centered by subtracting its vocabulary mean; let `D` be
the similarly centered clean lexical donor-minus-recipient mask-logit vector.
Report `dot(V,D)/dot(D,D)`, the cosine, `||V||`, the norm of the residual
`V - (dot(V,D)/dot(D,D))*D`, and TV. Dot products and squared norms use
coordinate-order `math.fsum`; report projection as null with a reason if
`||D|| < 1e-12`, and cosine as null if either norm is below `1e-12`. Do not
drop that case or change a threshold. Compare lexical, contextual and both
orthogonal axes. This makes output magnitude and alignment visible, but neither
alone identifies information transfer.

For every block-0 ingredient ± pair and each captured block-3 position, form
each odd downstream coordinate as `A_i=math.fsum((H+_i, -H−_i))/2` and
each even coordinate as `E_i=math.fsum((H+_i, H−_i, -2*Hclean_i))/2`.
Let `D` be that
position's clean same-template lexical donor-minus-recipient vector. Retain
both raw `H+` and `H−` bits and report the odd/even vectors, `||A||`,
`dot(A,D)/dot(D,D)` and cosine, with the same `1e-12` null rules. The
prespecified propagation contrast is lexical projection minus the mean of the
absolute projections of the two matched orthogonal controls; also report the
cross-template projection and every raw value. A positive contrast is evidence
of a donor-aligned response at the measured later site under these particular
perturbations, not evidence of a unique attention path.

For each stem and template, average the two recipient directions with
`math.fsum`; report every row and the six-stem means for both templates and
their paired difference at each site. If a required value is null, its
two-recipient aggregate is null with the missing count and reason; if any
required stem aggregate is null, the corresponding six-stem mean and paired
difference are null with the missing count and reason. Never silently average
only available cases. Preserve negative, tied, null and
ineligible clean baselines; the earlier sweet/sour top-five eligibility rule
is descriptive only. Report baseline full-vocabulary target probabilities and
top five, both signs, both orthogonal controls, self/final controls, achieved
norms and all whole-output summaries. No significance test or universal
generalization is prespecified.

## Execution boundary

A trusted outer operator exclusively creates and verifies a new absolute,
empty bytecode prefix before starting the isolated `-I -B` child, keeps it
quiescent and checks it after the child exits. The child rejects a missing,
symlinked or nonempty prefix before importing project modules. The interpreter,
startup/site machinery, installed dependencies and host remain trusted.
Require a separately observed direct-process exit and independent closed-record
audit before accepting a result. Recorded hashes and process observations do
not authenticate execution chronology, unmodified dependencies or whole-tree
cleanup.
