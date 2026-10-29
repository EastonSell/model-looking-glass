# Public DistilBERT taste-context matched-dose follow-up v1

## Question and public cases

For six new public sugar/lemon sentence pairs, how does a matched-L2 change at
the ingredient-token output compare with a matched-L2 change at the unchanged
`the` token output, in each encoder block and both transfer directions? These
are convenience frames for one checkpoint, not a held-out confirmation or a
test of a general taste concept. The complete texts, IDs, masks and prior
public-inventory digests are fixed in
`experiments/bert_taste_matched_dose_public_cases_v1.json`. The model and
tokenizer are the pinned offline DistilBERT revision used in the prior public
taste exploration. Execution uses CPU float32, one intra-op and inter-op
thread, deterministic algorithms and seed 20261030. Unknown local and
pretraining exposure is not certified.

Each pair differs only at ingredient position 7. The mask is at 4 and `the`
is at 1; all rows have length 10 and an all-one mask. The cases are coffee,
milk and soup with `after adding`, and cake, pudding and cocoa with `with
some`. These are six noun pairs using only two template types, not six
independent templates. All twelve exact rows are checked against the cited agreement v6,
matched agreement and taste v2 public inventories for exact or either-way
complete-row-prefix matches before any forward.

## Fixed design

At each of six block outputs, capture the clean ingredient and `the` vectors
for both rows in sugar-then-lemon order, ingredient then `the` per row. For
each recipient in sugar-then-lemon order, perform the full ingredient donor
replacement, ingredient self-patch, `the` self-patch, matched ingredient
interpolation and matched `the` interpolation, in that order. The donor is the
opposite ingredient row in the same frame and at the same block and token
position. Both self-patches use the same invocation's captured vector.
Perform a fresh clean forward for both rows at the end of each frame. With no
failed prerequisite this is 516 forwards: six frames, six blocks, fourteen
forwards per block, and two final checks per frame.

For each frame and block, compute Euclidean L2 distances between the two
clean ingredient vectors and between the two clean `the` vectors, using the
exact Python expression `math.sqrt(math.fsum((a - b) ** 2 for a, b in
zip(left, right)))` on binary64-decoded coordinates. The common radius is the
smaller distance. For each direction
and site, calculate the interpolation fraction as common radius divided by
that site's distance, then form `recipient + fraction * (donor - recipient)`
using separate binary64 subtraction and multiplication and `math.fsum` for
the final addition, then round once to float32 under nearest-even rounding.
Fractions must be finite and within
`[0, 1]`; no extrapolation or redrawing is permitted. Recompute each actual
float32 displacement using the same L2 rule. It must be nonzero and within
`1e-6 + 1e-4 * radius` of the intended radius. Record the distances, radius,
fractions, requested and delivered norms, relative errors and ratios to the
clean recipient-vector norm, and the raw recipient, donor and
submitted bits for both sites and directions. Each site and direction is
checked independently after float32 rounding. This tolerance permits about
10% relative error at the minimum radius, so delivered doses are not claimed
to be exactly equal. A nonfinite value, distance or
radius below `1e-5`, invalid fraction, or failed delivered-dose check makes
the attempt incomplete; retain already available records and do not silently
retry. The exact direct-child/process-tree and failure limitations of the
accompanying runner are part of the saved evidence.

If the clean recipient-vector norm is zero, its delivered-to-recipient norm
ratio is recorded as null with the `zero` reason; this does not itself exclude
a comparison. Denormal-preservation probes run before and after each model
call and around interpolation construction, and scoring is bracketed by the
same checks.

The comparator is a contextual `the` vector: bidirectional context may carry
ingredient information there. Matching norm within tolerance addresses only the intervention dose,
not its direction, distribution or downstream sensitivity. Interpolation may
still be off the learned manifold. Radii are set independently by frame and
block, so cross-block effect comparisons are descriptive.

The score is the mask logit for `sweet` minus the logit for `sour`. An
intervention effect is its recipient's patched score minus its own clean
score. A frame is eligible for the primary descriptive comparison exactly
when the clean sugar score is positive, the clean lemon score is negative,
and at least one of `sweet` or `sour` appears among the top five vocabulary
predictions for each row. Retain all ineligible, tied, negative and failed
outcomes; eligibility never gates model forwards. For each block and
direction, report each eligible frame's donor-directed matched-ingredient
effect minus its donor-directed matched-`the` effect, and their arithmetic
mean when at least one frame is eligible, using `math.fsum` in manifest frame
order divided by the number of eligible frames. With no eligible frames the
mean is null and its count is zero. Sugar-to-lemon uses the effect as
written; lemon-to-sugar negates it. Also report all raw effects and controls
for every frame, including ineligible frames, with no confidence or population
claim. Full-donor effects and exact self-patches are separate references.

For top-five ties, rank by decreasing logit then increasing vocabulary ID.
Record clean full-vocabulary mask logits, top-five tokens, and full-vocabulary
`sweet`/`sour` probabilities. For every intervention retain both target-logit
bits, probabilities and top-five summaries; only the two target bits are
independently reconstructible unless the full patched logits are saved.
Record selected pre/post/source bits and action coordinates, actual model-call
IDs and masks, and the bitwise landing check. The whole-output unchanged-token
check is retained as a hash plus an in-process assertion. Self-patches, final
clean forwards and all interventions at the final block must preserve the full
mask-logit bits. The public frames and this protocol were assembled after the
prior exploratory result and are to be published together before any new
forward; this is not a prospective holdout. A run does not become complete solely because a child wrote a
result: an independently observed clean direct-child exit, matching hashes,
no incomplete marker, and the observer's ordered journal, input, metadata,
selected landing, control and basic delivered-dose checks are also required.
An external operator must also observe and retain the parent process's own
zero exit. A parseable `complete` receipt alone does not prove that its write,
flush or fsync returned successfully; a nonzero or uncertain parent exit
leaves the attempt incomplete even if such bytes exist.
Before issuing `complete`, the observer also executes the independently
authored, source-bound standard-library reader from its checked source bytes
in a fresh isolated no-bytecode interpreter; that reader reconstructs the
result effects, interpolation and summaries. A separate replay of the same
reader follows the direct-child exit. These checks do not authenticate
execution, unchanged dependencies or whole-process-tree cleanup; using the
same reader twice is not two independent numerical methods.

Each journal contains `forward_start` and `forward_complete` records in the
fixed schedule, plus one `dose_plan` per frame/block. A failed dose
construction records a `dose_unavailable` event when possible. The entry
arguments are recorded by a temporary, read-only wrapper around the model's
forward method and checked against the intended IDs, mask and flags before the
original method is invoked; the method is restored after each call. Any
uncertain failure leaves the process and its partially saved data incomplete.
If a forward fails before the plan releases its captures, those partial
captures may be unavailable; prior completed records and the failed-forward
start remain in the journal when writing succeeds.

Run the published source through `python -I -B -X pycache_prefix=<fresh-empty-prefix>
scripts/bert_taste_matched_parent_v1.py` using an existing offline hub cache and
a fresh output path outside the checkout. A trusted operator must create the
exclusive parent prefix empty before interpreter startup and verify it remains
empty afterward; the parent separately creates the child's fresh prefix. The
child timeout is 900 seconds; a timeout is an incomplete attempt, not a
performance result. A separate `--preflight-only` invocation checks selected source
bytes, a fresh bytecode prefix, interpreter isolation flags and direct-child
exit without loading Torch, the tokenizer, cache or model. Actual runtime,
tokenization, inventory and checkpoint checks occur in the experiment child
before the first forward. The observer does not establish whole-tree cleanup;
on Windows its direct child may be a virtual-environment redirector. A timeout,
interruption or uncertain cleanup requires separate process reconciliation,
and no incomplete attempt is silently repeated.

The isolated child puts standard-library and installed dependency paths before
the checkout and restricts checkout imports to declared source files; it also
checks the origins of the already imported project modules. Its fresh prefix
prevents use of an existing conventional bytecode cache under that prefix.
This does not attest the complete bytes or behavior of Python, site-packages,
native dependencies or untrusted import machinery. Those remain trusted
runtime preconditions.
