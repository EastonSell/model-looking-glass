# Cue-state transfer across contexts: fixed CPU protocol

## Question and fixed public inputs

The earlier [two-dose study](https://github.com/EastonSell/model-looking-glass/blob/69018f04800b96f790bebe2d8077c0d4c8af1715/results/bert-cue-dose-continuation-cpu-v1-20261029/README.md)
found that its strict matched-control specificity rule was not met in the
six descriptively pooled evaluation cells at either 0.5 or 0.7. Its original
one-process attempt was incomplete. Here we ask a different, narrower question:
for the same recipient cue state, can a same-cue state from another lexical
context shift the `any`−`some` masked-logit contrast by at least half as much
on average as a different same-class cue state from that *same* donor context?
All compared patches in a context pair have the same requested input norm.

The fixed template and complete tokenizer rows are in
`experiments/bert_cue_context_transfer_cases_v1.json`. There is one discovery
pair, `nurse/checked/charts` versus `ranger/counted/birds`, and two evaluation
pairs, `librarian/sorted/books` versus `carpenter/built/shelves` and
`editor/revised/articles` versus `musician/played/songs`. In each context the
four cues are `never`, `rarely`, `always`, and `often`; the first two are
negative/low-frequency and the latter two are affirmative/high-frequency.
Each input has twelve tokens, with cue, mask and object at positions 4, 6 and
9. The six contexts and their pairings, all 24 complete rows, both discovery
and evaluation, are fixed before any new pretrained forward. The evaluation
texts were fixed before any follow-up output and are not used for post-output
tuning. They were visible during design, are public rather than blind, and are
not known absent from model pretraining.

An earlier tokenizer-only candidate was replaced before any new forward:
`mechanic/repaired/engines` and `teacher/graded/essays` exactly matched
previous factorial inputs, and the `never` variant of
`artist/painted/portraits` matched a prior negation input. The replacements
were chosen without new-model outputs. Before launch compare all 24 final
rows with the named available public BERT snapshots, in both prefix directions
with complete rows and with one terminal `[SEP]` removed. Any relation stops
the fixed attempt and is reported; no post-output replacement is permitted.
This named-snapshot check cannot establish private or pretraining nonexposure.
Two retained receipts record zero relations: one checks 1,061 row references
in 21 named public snapshots; the other checks 614 references in 14 named
snapshots plus one separately identified, fixed local planned color manifest.
The counts include repeated rows across inventories; neither receipt is a
complete exposure history.

The cue groups differ in frequency and temporal meaning; `rarely` is scalar
and is not equivalent to explicit sentential negation. `any` has free-choice
uses, and the two-token score is not a human acceptability judgment or a
general test of negation. BERT representations are contextual
([Ethayarajh, 2019](https://aclanthology.org/D19-1006/)); activation-patching
conclusions can depend on source/corruption and metric
([Zhang and Nanda, 2024](https://arxiv.org/abs/2309.16042)).
Here wording, donor-recipient context, state direction and dose all remain
possible explanations; a same-cue cross-context patch is an active
intervention, not a semantic null or an invariant-representation test.

## Intervention and controls

Use the cached `distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be` on CPU, float32, eager attention,
evaluation mode, one intra-op and one inter-op thread, deterministic
algorithms, no autocast or network, and seed `2026102906`. Hide CUDA before
Torch import and do not initialize it. Verify the pinned tokenizer/checkpoint,
source, versions and 24 inputs before model loading; record CPU environment and
before/after model-state hashes including nonpersistent buffers.

For each pair, first run all eight clean cue/context variants. Capture the cue,
object and mask states after block 2 (`cue2`, `object2`, `mask2`) and the cue
after final block 5 (`cue5`). For each recipient in context A or B and each
site, construct seven directed donors: the four cues from the *other* context,
then the other three cues from the recipient's own context. In each comparison
the same-cue and different-same-class donors share the other context, target,
site and requested norm. The two opposite-class donors are active controls;
the same-class and opposite-class donors from the recipient context make
within-context comparisons available. Both context directions are retained.

Compute all 224 required directed donor/recipient/site distances as
`sqrt(fsum((float(y_i)-float(x_i)) ** 2))` in coordinate order. Every distance
must be finite and nonnegative; otherwise the attempt is incomplete. Set one
pair-wide requested norm `q = min(0.5, minimum_distance)`. If `q < 0.05`, the
pair is dose-ineligible: retain eight clean forwards, distance evidence, and
eight final clean repeats, with no patches. A tie at 0.05 is eligible. The cap
0.5 is informed by the previous studies and can make effects weak; q is
computed by this fixed rule from the current pair's clean states and may vary
across pairs. Baseline-ineligible pairs still receive every dose-feasible
patch. Equal input norms do not equalize functional impact.

For a feasible transfer use binary64 `alpha = q / distance`, `0 < alpha <= 1`,
and separate binary64 subtraction, multiplication and addition followed by a
single nearest-even float32 cast. Retain the input, donor, submitted and landed
vectors, the full selected-site before/after states, full mask logits, q,
distance, fraction and achieved norm. Same-forward pre-action state must
bit-match the corresponding clean capture, landed bits must match the
submitted vector, and unselected token coordinates must remain unchanged.
The achieved norm must be positive and within relative error 0.001 of q.
Control, nonfinite, landing or partial-forward failures invalidate the attempt;
retain already returned bounded witnesses and a named phase where feasible.

Use fixed context A then B, cue `never`, `rarely`, `always`, `often`, and
site `cue2`, `object2`, `mask2`, `cue5` order. At each site, run the eight
self patches, then for each recipient the four alternate-context cues in cue
order and the three different same-context cues in cue order. Finally repeat
all eight clean forwards. A feasible pair has 8 clean + 32 self + 224 transfer
+ 8 repeat = 272 forwards; a dose-ineligible pair has 16. Three feasible
pairs require at most 816 forwards. Clean/repeat use passive capture hooks.
Self and repeat full mask logits must equal their clean outputs bit-for-bit;
every final-block cue transfer must preserve the entire clean mask-logit
vector. Object and mask sites are active transfer comparators, not nulls.

## Baseline, fixed outcome and full vocabulary

For each clean variant retain all mask logits, top-five vocabulary tokens,
`any`/`some` logits and log probabilities. Let `L=logit(any)-logit(some)`.
Compute `L` as a binary64 `fsum` of the two float32 logits; compute a patch
effect as `fsum((patched_L, -clean_L))`. Log probabilities use
`(logit - peak) - log(fsum(exp(value - peak)))` in binary64; top-five ties are
ordered by ascending token ID.
The pair's baseline is eligible only if all four clean
`L(negative)-L(affirmative)` gaps are strictly greater than 0.5 in *each*
context. Baseline failures and all their dose-feasible interventions remain in
the report. This outcome-selected operational screen is not a grammaticality
or population screen.

At cue2, each effect is patched-minus-clean `L` for the identical recipient.
For each of the eight ordered recipients let `s` be the signed effect from
the same cue in the alternate context, and `c` the effect from the other cue
in the same class in that same alternate context. Record all signed `s`, `c`,
their absolute values, opposite-class effects and same-context controls. Set
`A = fsum(abs(s_i))/8`, `B = fsum(abs(c_i))/8` in recipient order, and
`H = fsum((A, -B/2))` in binary64. A pair is informative for this fixed
comparison only if baseline and dose are eligible and `B >= 0.01`; its result
is `at_least_half` when `H >= 0`, otherwise `below_half`. Thus ties satisfy
the half-size inequality, and same-cue effects larger than the comparator also
pass. A small comparator is `uninformative`, not a negative result; preserve
its actual effects. This ratio-free rule and the 0.01 floor are operational,
not significance thresholds or evidence of directional agreement.

The discovery pair is reported separately and does not alter this protocol or
select evaluation cases. The evaluation aggregate requires both fixed pairs
to be baseline/dose eligible and informative. If either is not, the aggregate
is `unevaluable`; otherwise pool the 16 ordered recipient magnitudes, compute
`A_all=fsum(abs(s_i))/16`, `B_all=fsum(abs(c_i))/16`, and
`H_all=fsum((A_all, -B_all/2))`. It is `at_least_half` when `H_all >= 0`,
otherwise `below_half`. Report each pair and each of the 16 effects as well
as q and the paired denominators; no imputation or outcome-based retuning.
The pooled inequality weights pairs by their comparator magnitudes: one pair
can drive the aggregate while the other falls below half. Report pair-level
decisions separately and do not infer that both satisfy the aggregate rule.
The two evaluation pairs and shared template are a small descriptive sample,
not independent draws or a test of a population effect.

For each transfer at every site also report full-vocabulary total variation
from the recipient's clean mask distribution and a second TV after deleting
`any` and `some` then renormalizing. For each distribution, convert finite
float32 logits to binary64, subtract the maximum among retained logits, apply
`math.exp` in vocabulary order, divide by `math.fsum` of the weights, and
compute half `math.fsum(abs(p_i-q_i))`. Keep full logits for independent
replay; report the clean and patched probability mass on `any`+`some`
alongside every excluded-TV value, since renormalizing a small remaining mass
can amplify conditional TV. Report full/excluded TV by donor and recipient and the corresponding
same-cue/same-class averages per pair and site using `fsum` of the eight
ordered recipients divided by eight. TV is descriptive and is not a separate
prespecified success test. Differences in minimum q or donor context mean
these results cannot identify one isolated syntax, cue-class or semantic cause.

## Evidence and limits

Freeze the protocol, case manifest and exact executable and have an
independent reviewer inspect them before the first pretrained forward. Save
ordered, sync-flushed gzip JSONL on the E: scratch volume, including full
logits, geometry and control witnesses. Bound an individual serialized event
to 2 MiB and the compressed journal to 512 MiB; exceeding either bound stops
the attempt as incomplete. Require at least 2 GiB free on the
output volume and at least 1 GiB on C: at prelaunch; before each event require
at least 1 GiB on both volumes. These are point observations, not reservations.
Record incomplete attempts and directly waited exit status; do not retry or
replace cases based on outcomes. Independent saved-byte replay must check
schedule, dose, landings, controls, baselines, both outcome metrics and all
eligibility decisions. Source and saved records rely on trusted local execution
and are not execution or pretraining attestations.
