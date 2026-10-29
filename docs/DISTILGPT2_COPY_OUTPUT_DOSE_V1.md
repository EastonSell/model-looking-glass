# DistilGPT2 copying: family-matched projection-output dose

## Question and discovery history

The completed per-prompt input-dose study found the prespecified head-2
versus head-10 advantage in 4/8 families, below its 75% criterion. Its
equal input-slice norms nevertheless produced different mean `c_proj` output
changes. On eight new fixed public word families, this study asks whether
head 2 produces a more negative repeat-versus-shuffle target-log-probability
effect than head 10 when their *actual projection-output* L2 changes are
matched across both heads and all four variants of a family. Head 7 is a
single fixed same-layer active comparator; comparisons involving it are
descriptive. The previously observed input-dose study remains unchanged.

The prediction is `C = D_h10 - D_h2 > 0`, where `D_h` is the change in
target log probability on the repeated prompt minus the mean change on its
three shuffled variants. The output-dose primary criterion is met only if
at least six of eight families are evaluable, the mean C is strictly positive,
and at least 75% of evaluable C values are strictly positive. Otherwise it is
not met, or ineligible when fewer than six families are evaluable. This is a
fixed descriptive rule, not a significance test. A separate same-prompt
input-dose comparison is secondary and uses the same rule and its own
evaluable denominator. For each scheme, a family is evaluable only when all
four corresponding prompt plans and measurements pass. Neither comparison
selects cases from edited outcomes.

These new public families are held out from the previous analysis, but were
selected after prior outcomes were known; they are not a blind sample or
evidence that words or formats were absent from model pretraining. No claim of
a unique copying mechanism or population-wide generalization follows.

## Fixed public cases

Use the pinned DistilGPT2 checkpoint and GPT-2 tokenizer from the input-dose
study. The eight ordered candidate pools below are fixed before tokenization
or any new model forward; spaces separate candidate words.

1. `valley river stream lake pond field plain cliff coast shore bay island desert mountain forest prairie marsh dune delta harbor lagoon reef volcano hillside plateau ravine tundra oasis beach jungle basin meadow canyon`
2. `steak chicken turkey pork beef cheese butter cream soup salad onion carrot cabbage pepper tomato rice corn wheat biscuit cookie pudding fruit apple pear peach plum lemon lime grape melon cherry berry lettuce olive potato bread muffin cereal bacon honey pasta yogurt apricot mango`
3. `dog cat horse cow sheep goat pig mouse rat squirrel turtle frog snake eagle hawk owl crow duck goose swan crab lobster shrimp whale dolphin lion tiger bear fox wolf deer moose raccoon badger otter seal shark trout robin parrot insect ant bee butterfly salmon lizard rabbit`
4. `brush broom bucket ladder rope chain wheel pipe basket pot pan knife spoon scissors tongs shovel screwdriver pliers drill axe saw chisel hammer wrench rake spade scissor nail screw bolt nut wire pulley lever needle thread`
5. `guitar piano violin flute clarinet trombone trumpet cello whistle singer dancer concert stage theater melody rhythm song choir jazz rock drum harp banjo bass cymbal saxophone accordion opera orchestra festival chorus composer quartet music tempo harmony soprano`
6. `train bus truck taxi car motorcycle wagon ferry submarine rocket satellite glider balloon metro subway rail truck airplane bicycle ship boat van sedan helicopter scooter carriage canoe kayak barge cruiser tram minivan sailboat spaceship hovercraft vessel cable`
7. `shirt coat jacket sweater dress skirt socks shoes boots gloves hat cap belt purse wallet watch button fabric cotton wool silk scarf mittens trousers sandal pajama robe bracelet tie necklace ring buckle zipper sleeve collar apron brooch pendant pocket ribbon cardigan`
8. `bed pillow curtain mirror shelf carpet towel soap cup plate bowl glass bottle box bench stool dresser closet carpet chair table desk sofa couch lamp clock frame cabinet window drawer stove oven kettle skillet fork blanket sheet mug rug vase`

For each pool in this order, tokenize each `" " + word` without special
tokens and record the whole candidate/exclusion log. Retain the first
occurrence of a repeated word and exclude later occurrences. Exclude
non-single-token words, IDs used by `PUBLIC_COPY_GROUPS`,
`PUBLIC_EVALUATION_GROUPS`, `FOLLOWUP_GROUPS`, `FACTORIAL_GROUPS`, and
`PAIRS_DOSE_GROUPS` from the public v6, v7, v9, v12 and v15 exposure modules,
the 48 selected IDs in `experiments/distilgpt2_copy_head_prompt_dose_cases_v1.json`, and IDs already
selected for an earlier new family. Scan ordered six-word combinations in
lexicographic pool-position order; choose the first whose IDs are distinct,
whose four exact input token sequences and one-token target continuations
retokenize as specified below, and whose inputs and continuations have no
bidirectional full-row-prefix relation with any known public input or saved
continuation, the fixed 128/256 public support rows, or any previously chosen
new input/continuation. Record every rejected combination and reason. If any
pool has no valid combination, stop as construction-ineligible; do not select
using model outputs or replace a frozen case.

For selected IDs `a,b,c,d,e,f`, the four six-token first blocks use index
orders `(0,1,2,3,4,5)`, `(4,2,0,3,1,5)`, `(1,3,4,2,0,5)`, and
`(3,0,4,1,2,5)`. Append `a,b,c,d,e` to each as the 11-token actual forward
input and score `f` as the single-token continuation. Reserve the 32 actual
inputs in a current public exposure successor before forwarding, and compare
the 32 score continuations as metadata, not additional model exposures.
Check current public inventory at launch; an exact own-reservation exemption
must bind identity, prompt, target and status. A scoped private custodian
comparison is recorded only if independently authorized and available; this
public study does not inspect protected rows or claim unknown-pretraining
disjointness.

## Fixed interventions and outcomes

Use layer 3 `attn.c_proj` input at final position 10, heads 2, 10 and 7,
each 64 of 768 channels. For each family first save clean and bit-identical
no-op outputs and full projection-input/output witnesses for all four prompts.
Check the checkpoint `c_proj.weight` has Conv1D layout `(768,768)` and retain
the exact float32 rows `64*h:64*(h+1)` for each head with their hashes. An
independent reader compares those bytes to the pinned checkpoint, not merely
to an extracted-slice hash. Let
`x[v,h]` be its no-op slice and `W_h` those 64 input rows. In binary64, in
fixed index order with `math.fsum`, compute `g[v,h] = x[v,h] @ W_h` and
`n[v,h] = ||g[v,h]||_2`; the affine bias cancels from a difference.
The family output dose is `q_out = min(n[v,h])` over all 12 pairs. H7 or
any one variant can therefore constrain all arms; report which one did.

If finite `q_out <= 2^-8`, mark the output family dose-ineligible, retain
all no-op witnesses and norms, and skip its output-dose edits. Otherwise set
`f_out[v,h] = q_out / n[v,h]`, compute `(1-f_out)*x[v,h]` in binary64,
round each coordinate once to float32 and freeze the exact delivered bits.
Before any edited forward, compute the realized head-input removal and its
predicted projected change vector from these bits and `W_h`; if any predicted norm
differs from `q_out` by more than `1e-4*q_out` or their range exceeds that
amount, mark output dose-ineligible and skip all its output-dose edits.

Independently plan the old per-prompt input-dose control on the same no-op:
`q_in[v] = min_h ||x[v,h]||_2`, with the previous `2^-20` floor and `1e-5`
relative landing tolerance for both each head's error and the within-prompt
head-norm range. Its three arms are secondary; input-dose floor
or rounding failures do not remove a valid output-dose arm, or vice versa.
When both are eligible, execute six active arms per prompt in a fixed
left-rotation of `(out_h2,out_h10,out_h7,in_h2,in_h10,in_h7)` by
`(4*(family-1)+variant) mod 6`, followed by clean-after. If only one scheme
is eligible, filter this six-arm rotation without changing the remaining
order. Run every preplanned arm even if an earlier actual output norm misses
tolerance; classify the affected family afterward. A skipped arm is
recorded; even identical planned bit patterns are separate recorded arms.
The maximum is 288 forwards and 80,604,288 bytes of per-action tensor data,
plus 589,824 bytes for the three weight slices and bounded metadata.

For every active arm, require the complete pre-action input witness to match
that prompt's no-op bitwise, the intended float32 head slice to match planned bits,
and every other input coordinate to remain unchanged. Retain the entire
`c_proj` output and require all other token positions to match the no-op.
For each output-dose arm, compute the actual final-position output difference
against its no-op in binary64; all 12 norms must differ from `q_out` by at
most `1e-4*q_out`, and their range must be at most that amount. A correctly
landed, finite plan whose computed norm or range misses its tolerance makes
only the relevant comparison dose-ineligible, retaining all arms and raw
evidence. For output-dose arms only, also require each actual output-difference
vector to be within `1e-4*q_out` L2 of the checkpoint-derived predicted vector;
a finite miss is likewise output-dose-ineligible. This gate never invalidates
the separate input-dose comparison. Any wrong planned-slice bit, unexpected other input or output bit,
missing/nonfinite data, or uncertain
hook cleanup, restoration or process failure makes the attempt incomplete.
Do not tune fractions or tolerance from model target outcomes.

Retain actual input-removal L2, predicted and actual output L2 and output
difference vector (edited output minus no-op), pairwise output-direction cosines, target logit, target
log probability/rank/top and tie counts, full-vocabulary KL(clean||arm) and
total variation for each arm. Report clean target ranks and each family's
clean repeat target log probability minus the mean of its shuffled target log
probabilities as descriptive baseline behavior, with no baseline eligibility
gate. Compare output-dose and input-dose outcomes
only descriptively: their interventions and dose constraints differ. Equal
projection-output norms do not match output directions or downstream nonlinear
gain. A pairwise output-direction cosine is unavailable if either vector's
actual L2 norm is at or below `2^-8`.

The source, exact cases, protocol, weights/tokenizer identities and manifest
must be public and independently reviewed before the first new forward.
Use the existing checkpoint without copies, one CPU owner, offline mode,
prescribed source/runtime/full-state and buffer checks, direct process
observation and a fresh bytecode prefix. Use E: for the attempt and require
the fixed tensor bound plus 16,777,216 bytes of metadata headroom and a
1,048,576-byte failure reserve there before launch; require at least 1 GiB
free on C: for temporary/cache headroom. Never retry an uncertain
or incomplete fixed attempt. Publish all positive, negative or ineligible
results and partial failures; saved-byte replay and cooperative process/clock
records are not independent execution attestation.
