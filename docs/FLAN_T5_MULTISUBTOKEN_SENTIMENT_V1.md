# FLAN-T5 sentiment study with multi-subtoken adjective sites, v1

This is a new, outcome-informed protocol, not a revision or replacement of
the closed 1/12 selection-incomplete study. Publication fixes the hypotheses,
templates, selection and decision rules below before any new case selection,
painting-context tokenization or model forward. The separately reviewed
selector, exact case manifests and adapted execution source must be published
before the first model forward. No successor case has been selected and no
painting-context row has been tokenized when this protocol is authored.

## Evidence and question

The [released prior screen](https://github.com/EastonSell/model-looking-glass/tree/6b7433fba2c909c8f383565570bfc0844b77f2ad/results/flan-t5-sentiment-intervention-v1-20261029)
selected only one of twelve fixed units before any reported model forward. A
later [tokenizer-only development pilot](https://github.com/EastonSell/model-looking-glass/tree/82a37efd72f64893f27c30fae2ff09c963f4a8b5/results/flan-t5-multitoken-feasibility-v1-20261029)
used 160 proposed adjectives in four fixed templates. It found structurally matched buckets that can contain at most
three four-word units each at two and three adjective subtokens, and one at
four subtokens, after checking only the two named earlier public sources.
These are upper bounds before independent semantic review and do not imply
the model supports the task or that all previous exposures are known.
In the earlier released, semantically accepted convenience units, 19/20
positive adjectives had one lexical subtoken versus 6/20 negative adjectives;
this small exposed pool is a warning about confounding, not a population rate.
The general sensitivity of patching conclusions to the intervention and
metric is discussed by [Heimersheim and Nanda](https://arxiv.org/abs/2404.15255)
and [Zhang and Nanda](https://arxiv.org/abs/2309.16042); neither validates
this study's numeric thresholds or establishes a unique mechanism.

On the pilot's two tokenizer-exposed review subjects `story` and `concert`
and the two established instruction frames,
does an equal-L2 move of the **last encoder token whose offset overlaps the
adjective's characters** toward an opposite-polarity donor move the complete
positive-versus-negative label score more in the donor direction than four
active controls? This tests one specified final-subtoken intervention. It is
not a full-word replacement, a unique sentiment representation, a protected
lexical holdout, or a claim about unobserved model reasoning. A separately
reported transfer test uses the same preselected words in a fixed third
subject, `painting`, not tokenized in the 640-row pilot. It asks whether the
predeclared effect persists in this locally new context; the words remain
tokenizer-exposed and no lexical or pretraining independence is claimed.

The candidate lexicon was selected and tokenized after seeing the earlier
failure. This distinct study intentionally selects from that 160-word pilot:
its words are tokenizer-exposed development material, not a lexical-blind
pool. These exact `story`/`concert` and instruction rows were already
tokenized in the pilot. They may be called locally model-unscored only within
the documented source scope. Before any new forward, freeze a separate exact
case manifest, source revision, active pool/custodian scope, donor mappings,
numerical construction, seed, primary rule and resource bounds. Unknown
pretraining and unrecorded exposure remain unknown. A genuinely fresh lexical
holdout needs a separately frozen follow-up. Do not reuse the original failed
12-unit pool as replacements for that study.

## Candidate selection and controls for segmentation

Before semantic ratings, apply only the fixed saved-pilot filter: retain
candidates in a `1 lexical/1 boundary` or `2 lexical/2 boundary` signature
bucket capable of a 2-positive/2-negative unit after the pilot's two public
source exclusions. Record all 160 candidates and every filter or exposure
reason. The saved pilot has 47 and 21 candidates in these eligible strata,
before any additional current-source exclusions. This prefilter uses no
model outcome and does not establish semantic eligibility.

Have two independent reviewers rate each retained word's ordinary positive,
negative or ambiguous meaning in the two exact review sentences, `The story
was {word}.` and `The concert was {word}.`, before observing model outputs.
These ratings apply to both instruction frames, which present the same review;
instruction-following differences remain part of the measured behavior.
Give each reviewer the words in alphabetical order, with fixed `story` then
`concert` order, omitting proposed labels, original list order and token
signatures; the original list order would reveal proposed polarity. Record
each reviewer's prior exposures, instructions and ratings, then map the
ratings back to the frozen candidate order only after both finish. Blinding
cannot undo prior exposure. Retain both reviews and reject disagreements or
ambiguous cases for the relevant context, with reasons. A candidate is
accepted for selection only when both reviewers agree with its proposed
polarity in both review sentences.
Keep every semantic rejection. The two same-polarity words
need not be synonyms, and opposite words are not presumed equally intense.
Retain the pilot's 43 scoped old-source overlaps and all other exclusions in
the development ledger. Additional applicable public or protected source
comparisons are performed by the relevant custodians; no private rows need
to be revealed. A future protected lexical pool should exclude the pilot's
entire candidate inventory rather than treating it as blind.

At minimum, bind the ten revision/SHA-pinned source descriptors in the
[released prior screen](https://github.com/EastonSell/model-looking-glass/blob/6b7433fba2c909c8f383565570bfc0844b77f2ad/results/flan-t5-sentiment-intervention-v1-20261029/custody/exposure-descriptors.json),
including the 24-row weight proposal and known-injection/localization rows,
and the released prior sentiment pool itself. Refresh the relevant public and
custodian scope before selection and publication, preserving source identities,
missing-scope limitations and comparison receipts. Against each applicable
recorded complete token row, remove exactly one terminal EOS from each valid
row and check both prefix directions. Do this for discovery candidates before
selecting slots and for the selected painting rows before model work; a
painting collision leaves that slot unmeasured, never replaces discovery.
This finite scope does not establish global or pretraining novelty. The pilot
is intentionally reused as development input, not excluded as a lexical pool.

Re-tokenize every remaining proposal in the actual pilot contexts before
selecting any unit, recording the complete ordered screen. Identify a
candidate's word span from every token whose nonempty offset overlaps the
adjective plus its immediately preceding ASCII space; then require every such
token to be attended, nonspecial and nonunknown. In particular, a standalone
whitespace token belongs to this boundary span. Require the recorded offsets to cover that character interval exactly,
without gaps or inclusion of punctuation, and to occupy contiguous token
positions; offsets in that span must be ordered, nonoverlapping and adjacent.
The intervention site is the **last token whose offset overlaps
an adjective character**, which is not necessarily the whole boundary span.
Keep both the lexical-subtoken count and the boundary-inclusive count.

For each subject/instruction group compare the complete token IDs, attention
masks and special masks outside each candidate's full boundary span. Build
disjoint four-word units `(P1,N1,P2,N2)` from independently accepted words
with matching lexical count, boundary-inclusive count and final adjective
site index in both instruction frames and both subjects, and identical
outside-span sequences in each group. Require equal lengths and all-one source
attention masks within each four-row group, with exactly one final EOS.
Check those facts directly on the exact frozen cases; a signature match in
the pilot's other contexts alone is not sufficient. Require the four complete
word-span token-ID sequences to be distinct, but do not require the four
final-token IDs to differ. The proposed denominator
is four word units: two with one lexical and one boundary-inclusive token,
then two with two lexical and two boundary-inclusive tokens. The same four
words in a unit appear under both fixed subjects and both instructions. The
two subjects are fixed above; the pilot-list ordering is fixed in its saved
candidate file and predates any new model scoring.
Process unit slots 0 and 1 for the one-token stratum, then slots 2 and 3 for
the two-token stratum. For each slot, enumerate `(P1,P2,N1,N2)` tuples in
lexicographic order of their indices in the pilot's original respective
polarity lists, requiring `P1 < P2` and `N1 < N2`. Select the first tuple
whose words are unused, accepted in every context, pass the applicable
exposure checks, and satisfy every exact group constraint above; mark all
four words used. If none exists, keep that slot empty and continue to the
next fixed slot. Record every candidate and disposition and the deterministic
search outcome. Units are never replaced after any new pretrained output.
If a stratum underfills, retain and evaluate any available units and report
each missing slot as unmeasured against the four planned slots; do not fill
it from a different stratum or label it a failed intervention.

The third-context subject and exact templates are fixed here before any
tokenization of that context:

```text
Classify the sentiment of this review as positive or negative: The painting was {word}.
Is this review positive or negative? The painting was {word}.
```

Select and seal all four discovery slots solely from the story/concert
screen. Then, and still before any model forward, tokenize up to eight words
from any selected slots 0 and 2 in these two painting templates, preserving
every row and rejection. Have the same two independent reviewers rate the exact
`The painting was {word}.` sentences in alphabetical order without proposed
labels or token signatures, applying each judgment to both instruction
frames. Check applicable exposure scopes, the same per-group token/mask/site
constraints, both reviewers' polarity agreement, and the prescribed `1/1`
stratum for slot 0 and `2/2` stratum for slot 2. An absent discovery
slot or a painting semantic, exposure or tokenization failure leaves that
holdout slot unmeasured; do not substitute another slot or word. These
holdout checks cannot alter discovery selection. The same reviewers cannot be
blind to their own earlier story/concert ratings; record that exposure. Run all valid painting arms
under the same rules, retaining unsupported and failed rows. Treat the two
planned holdout units as a separate, small context-transfer assessment.

Do **not** require four distinct final-subtoken IDs as a proxy for distinct
words: different multi-token adjectives may share a final token. Record each
full word's token IDs, final ID, count and absolute index. Prespecify
descriptive strata for opposite- and same-polarity donor pairs sharing the
final token ID versus differing at that ID. The existing pilot can form
opposite-polarity shared-final-ID pairs but no four-word unit with all four
words sharing one final ID, so imposing that rule would underfill it.
Matching IDs and positions still does not match the earlier subtokens,
semantics, encoder context or downstream effect.

## Intervention and readout

Use `google/flan-t5-small` revision
`0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` and seven-file asset manifest
SHA-256 `d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083`,
with final normalized encoder output as in the original protocol. Preserve the baseline encoder state and
attention mask of **each recipient**. Map opposite donors `P1↔N1` and
`P2↔N2`, same-polarity donors `P1↔P2` and `N1↔N2`, all within the same
subject/instruction group. Patch just the recipient's final adjective-overlap
row toward the corresponding donor's own final adjective-overlap row.
Earlier adjective-subtoken rows remain the recipient's, and a donor state can
encode its whole sentence, not just polarity.

Retain the original three active controls: same-polarity donor at that
adjective site, and opposite-polarity donor at the subject and at instruction
position 0. Define the subject site as the last attended, nonspecial token
whose offset overlaps a subject character; position 0 must also be attended
and nonspecial. Verify that every adjective-character-overlapping token,
including the patched last token, is distinct from both the subject and
instruction sites in every case. The subject and instruction sites must also
be distinct. Each intervention modifies only its selected recipient
row. Add a fourth active control at the adjective: a fixed-seed
Gaussian direction orthogonalized against that receiver's opposite-donor
vector and normalized to the same dose. It is one direction, not a
distribution or universal null. For unit `u`, subject `s`, instruction `i`
and receiver index `j` (all zero-based), instantiate Python 3.11.15
`rng = random.Random(20261029 + 24*u + 8*s + 4*i + j)` once and call
`rng.gauss(0,1)` 512 times to draw binary64 values `g` in order. For natural
opposite vector `d`, compute `c = math.fsum(g[k]*d[k] for k in range(512)) /
math.fsum(d[k]*d[k] for k in range(512))`, then `g_perp[k] = g[k] - c*d[k]`
and `n = math.sqrt(math.fsum(g_perp[k]**2 for k in range(512)))` using
separate binary64 operations in that order. At radius `r`, separately compute
`base[k] + (r/n)*g_perp[k]` and round once to binary32. This raw random
vector's norm does not enter the group minimum q. Save the seed, runtime,
all 512 binary64 source values or exact binary64 tensor bytes, intended
direction, numeric-mode probes and float32 landed bits. A digest alone does
not permit reconstruction. Construct it once for both doses; do not
resample. If its projected residual is zero or nonfinite, or its rounded
landing misses the required tolerance, mark that orthogonal comparison
unavailable and keep the valid donor arms. The four-control primary is
unmeasured if its high-dose orthogonal arm is unavailable. Do not silently
substitute a new seed or direction.
Here `u` is the original unit slot, `s=0` for story, `s=1` for concert and
`s=2` for painting, `i=0` for the declarative instruction and `i=1` for the
question instruction, and `j` indexes `(P1,N1,P2,N2)`. Empty unit slots keep
their indices. This stride assigns a distinct seed to each planned receiver.
Compute all 16 natural donor-to-recipient site distances and set each group's
radius to their minimum before constructing any matched or orthogonal
direction. A nonfinite distance makes the attempt incomplete;
zero or `q/2 <= 1e-6` makes that group's matched-dose comparison ineligible
without dropping the offending distance: do not divide by a zero direction or
draw/project matched landings, but run finite natural anchors and plumbing
controls. Use the predeclared `q/2` and `q`
levels and match the achieved L2 norm within a receiver across the original
four arms and the orthogonal arm. Form each donor direction by subtracting
the receiver's float32 row from the donor's after binary64 promotion; compute
each distance as `math.sqrt(math.fsum(delta[k]**2 for k in range(512)))` in
dimension order. At
nominal dose `r`, calculate `base + (r/distance)*direction` in that written
order using separate binary64 operations and round each coordinate once to
IEEE binary32, round-to-nearest ties-to-even. Construct every planned vector
before that group's matched-dose actions and verify the actual delivered bits.
Record natural distances, common q, float32 landed bits, realized norms,
direction cosines, untouched-state checks and failed landings. For nominal
dose `r`, require each achieved norm within
`min(max(1e-5, 1e-5*r), 0.01*r)` of `r`, each active control within 1% of
the receiver's opposite-site achieved norm, and cosines to the intended
directions at least 0.99 for each of the four donor-derived arms. The
orthogonal arm is checked separately against its own intended direction and
against the opposite direction below. Finite norm/cosine misses make the affected arm unavailable at
that dose; skip that invalid landing and run the other valid preplanned arms.
A dose is checked independently: a low-dose-only miss does not make a valid
high-dose primary unmeasured. At each dose first check each arm against its
own nominal-radius and direction tolerances. If the opposite-adjective
landing is valid, additionally apply the 1% relative norm check to each
control. If that reference landing is unavailable, including zero achieved
norm, do not divide by it or call any control matched: record the relative
test as unavailable, but run other arms that pass their own finite nominal
checks descriptively. In either case retain all reasons and landed bits.
A group requiring an unavailable donor arm has its B3 and B4 comparisons
unmeasured at the affected dose; an unavailable orthogonal arm alone leaves
B3 measurable when all donor arms are valid. Wrong landed bits, nonfinite
states or uncertain execution stop the affected pass and make it incomplete;
retain partial records and do not silently retry the same frozen pass. Any
continuation is separately recorded and labeled with its already exposed
outcomes. Retain every valid result. For an
orthogonal arm also require the realized absolute
cosine to the opposite-donor direction to be at most `1e-4` and its realized
cosine to the intended orthogonal direction to be at least 0.99; a finite
miss leaves the three-control comparison available, but a high-dose miss
leaves the four-control primary unmeasured. Matched norms
do not make these controls functionally equivalent. A finite insufficient or
unmatchable dose is an ineligible comparison, not negative evidence; retain
its clean results and other valid controls.

Within each group use receivers in `(P1,N1,P2,N2)` order, dose `q/2` then
`q`, and at each dose run valid opposite-adjective, same-adjective, subject,
instruction and orthogonal arms in that order. A finite unavailable arm does
not remove a later valid arm. High-dose B3/B4 eligibility is evaluated from
the high-dose landings and comparisons, independently of low-dose status.

Run the unmatched full natural opposite-donor replacement at the final
adjective row for each finite receiver, regardless of baseline support or a
too-small matched q. Also run an unmatched natural whole-adjective reference
by replacing every token row whose offset overlaps an adjective character,
paired with donor rows by relative order; matching span lengths make this
well-defined. A standalone leading-space-only row would stay unchanged, and
every non-adjective row stays the recipient's. Record the whole-span composite
L2 norm and all outputs. For one-token units this is the same mathematical
intervention as the single-row natural swap: run both predeclared forwards
and require their patched-state bits and resulting full logits to agree. The
two natural anchors have
unmatched doses and cannot by themselves isolate distributed information;
an equal-total-L2 multi-row comparison is a separately frozen future test.

For plumbing, check clean encoder passthrough, per-site self patches and
full-state opposite-donor controls. A full-state donor control uses the
**donor state and donor attention mask**, and compares identical decoder
prefixes. The selected group's masks and lengths must already be equal; verify
that equality and explicitly pass the donor mask. This study does not promise
handling of unequal-length full-state donors.
These controls test the implementation, not selectivity. Record any mismatch
or uncertain execution as incomplete rather than a behavioral miss. Require
the receiver passthrough and every self patch to reproduce that receiver's
clean emitted IDs and every full decoder logit bit at identical prefixes.
Require each full-state donor patch to reproduce the corresponding donor's
clean emitted IDs and full logit bits at identical decoder prefixes, using
the donor state and its mask. Save the compared outputs. A mismatch or
uncertain cleanup immediately stops that affected pass as incomplete;
finite behavioral misses remain descriptive.

Retain all clean and measured intervention rows, including baseline-wrong
and baseline-unsupported cases. For every tokenizer-selected group with a
finite, landable matched dose, run **all** predeclared matched donor arms and
any valid orthogonal arm regardless of clean correctness. Baseline-unsupported
groups are descriptive and cannot enter the primary numerator. Save complete
greedy output IDs, EOS and full-vocabulary logits, and teacher-forced full
label sequences including EOS. Verify the pinned tokenizer encodes `positive`
and `negative` without added specials as `[1465]` and `[2841]`, then append
EOS ID 1 to get `[1465,1]` and `[2841,1]`; reject a mismatch. For each label,
teacher force with decoder inputs `[0] + label_ids[:-1]`, explicit all-one
decoder and source masks, and no KV cache. Save every full vocabulary logit
row and each target token's rank `1 + count(logit > selected_logit)`,
including EOS. Let `m = log P(positive + EOS) − log P(negative + EOS)`; report it as a
two-sequence log-probability ratio, retaining both teacher-forced token
sequences, decoder prefixes and raw vocabulary logits. Reproduce the prior
protocol's binary64 `math.fsum` of selected float32 log-softmax values; do not
infer a calibrated class probability. Greedy generation uses decoder-start
ID 0, explicit all-one source/decoder masks, no KV cache, sampling, forced
token or logits processors, raw lowest-ID argmax, at most four generated
tokens, and stops on EOS ID 1. Save each prefix and full raw logits; record
truncation, ties and every special token without suppressing them. Strict
bare-label scoring requires exactly one emitted terminal EOS, no other
special/unknown/pad/sentinel, and a label match after NFC, outer whitespace
stripping and case folding. Decode without dropping specials or tokenization
cleanup. This score is separate from generated labels and
full-vocabulary top tokens. Preserve the original donorward sign convention
`w = -Δm` for a positive receiver and `w = +Δm` for a negative receiver,
every per-receiver control effect, and changes at both `q/2` and `q`. Report
first-decoder-step full-vocabulary TV at the common decoder-start prefix:
for each logit vector obtain binary64 softmax probabilities after subtracting
its maximum, then `0.5 * math.fsum(abs(p[k] - p2[k]) for k in range(32128))`;
greedy prefixes after divergence are not equivalent conditioning contexts.
Label first-step distribution changes and greedy reversals independently; a
margin shift need not change generation.

In each subject/instruction group, clean baseline support requires all four
cases to generate the assigned strict bare positive/negative label with EOS
and no other special token, with a unique vocabulary maximum at every greedy
step through EOS, and all four complete-label `m` signs to agree strictly
with their labels. Ties and truncation do not count as supported. Retain and report ties and wrong or truncated
outputs. The fixed larger-dose group rule requires the mean opposite-site
donorward effect `A >= 0.10` nat, `A - B4 >= 0.05` nat, and at least three
of four donorward effects strictly positive. `B4` is the largest of the four
control kinds' mean absolute effects, including the orthogonal direction.
For opposite effects `w` in `(P1,N1,P2,N2)` order, compute
`A = math.fsum(w[j] for j in range(4))/4.0`. For each control's signed
effect list `c` in the same order compute
`math.fsum(abs(c[j]) for j in range(4))/4.0`, then take the maximum for B4.
Use binary64 operations and the Python binary64 constants `0.10` and `0.05`
for the boundary comparisons.
Also report the prior three-control `B3` rule as a secondary comparison. These
floors are prespecified operational thresholds, not calibrated practical
effects or significance tests. As a fabricated arithmetic check, opposite
effects `[0.16,0.20,0.12,0]` give `A=0.12` and three positives; orthogonal
effects `[0.12,-0.12,0.12,-0.12]` have mean absolute effect `0.12`, not zero.
With all other controls zero, B3 passes while B4 fails. Absolute values are
taken per receiver before averaging.

A word unit is evaluable only if all four subject/instruction groups are
baseline-supported, high-dose-matchable, have every required high-dose B4 arm and are
scientifically replayed. It passes only if all four meet the group rule. The
primary threshold is at least three passes among four planned slots. Once the
available schedule has completed and its evidence has been independently
replayed, let `E` be evaluable units and `K` their measured passes. Report
the separate counts `K` and `E`, the rate `K/E` (null when `E=0`), missing
reasons and the possible pass-count interval `[K,K+4-E]`.
The criterion is met if `K >= 3`, not met if `K+4-E < 3`, and otherwise
undetermined. Any `E < 4` classification is a partial assessment; empty
slots in the upper bound are hypothetical, not observed cases. Thus `K=2,
E=3` gives `[2,3]` and is undetermined; `K=3,E=3` gives `[3,4]` and is
met in a partial assessment; `K=0,E=2` gives `[0,2]` and is not met in a
partial assessment. Do not count missing as measured failures. An incomplete
or uncertain execution/exit, failed plumbing control, or missing saved
evidence in the relevant passes overrides that assessment's interval
classification: it is incomplete.
Keep the distinction between a measured rule failure,
baseline-unsupported and unmeasured controls. Report all 16 planned discovery group statuses,
all individual effects and cross-subject or token-length strata descriptively;
the reused words and contexts are dependent and two units per stratum cannot
support a population comparison. The second legacy instruction admits a
yes/no reading, so strict bare-label accuracy partly measures output format.

For the separate painting holdout, a planned slot is evaluable when both
instruction groups are baseline-supported, high-dose-matchable, have all high-dose B4 arms
and pass scientific replay; it passes only when both groups meet the same
fixed rule. The two-slot criterion requires both to pass. Report `K_h` and
`E_h` separately, the rate `K_h/E_h` (null when `E_h=0`), all four planned
group statuses and `[K_h,K_h+2-E_h]`: met if `K_h=2`, not
met if the upper bound is less than 2, and otherwise undetermined. Any missing
slot is a partial assessment, not a measured failure. A painting-pass process,
plumbing or evidence failure makes that holdout incomplete. The words are
reused discovery units with one unit per token-length stratum, so even a
positive transfer result is limited to these two planned units and context.
If no discovery unit is selected, run no model action and report discovery
selection-incomplete, with its `[0,4]` bound and all missing reasons; the
painting holdout is likewise selection-incomplete with `[0,2]`.

## Compute and interpretation

Adapt the existing sentiment runner for offline, eager CPU float32 execution
and the pinned FLAN-T5-small revision for this study: its earlier source does
not implement this protocol's arm and missing-data rules. The current shared GPU broker does not
support this decoder-state workload, and the supervised GPU observer cannot
yet handle every unsuppressed greedy prefix. Pin Python/runtime versions,
one intra-op and one inter-op thread, deterministic algorithms, and disable
KV cache. A GPU adapter is a separate follow-up and is not a prerequisite.
One supervisor runs a separately reaped direct child and journal per subject,
all with the same verified model identity, in fixed `story`, `concert`, then
`painting` order. Check intervention cleanup after each action. Each pass has
its own exit and complete evidence record; a later painting-child failure does
not erase already completed and independently replayed discovery passes.
Within a pass, process original unit slots in ascending
order and instructions 0 then 1. For each group record all four clean
receivers, construct and check its landings, then process receivers in
`(P1,N1,P2,N2)` order: passthrough, self patches at adjective/subject/
instruction sites, full-state donor, valid matched arms in the order above,
natural final-row swap and natural lexical-span swap. Repeat clean P1 once
after instruction 0 for each selected unit in each pass. Retain all
predeclared available actions even if the clean baseline is unsupported.
A nonzero or unknown exit or incomplete discovery-pass evidence makes the
discovery primary execution-incomplete; incomplete painting-pass evidence
makes the separate holdout incomplete. Preserve partial records.
At four units × two subjects × two instructions × four receivers, the
conservative schedule has 64 clean, 64 passthrough, 192 self, 64 full-state
donor, 640 matched-dose (five arms × two doses), 64 natural final-row, 64
natural whole-adjective and eight repeat output evaluations: at most **1,160
output evaluations** and **7,032 component forwards**. The two-slot painting
pass adds 16 clean, 16 passthrough, 48 self, 16 full-state donor, 160 matched,
16 natural final-row, 16 natural whole-adjective and two repeat evaluations:
at most **290 evaluations** and **1,758 component forwards**. Across all
three passes the maximum is **1,450 evaluations** and **8,790 components**, 
with at most four greedy and two teacher-forced decoder calls per evaluation
plus 90 clean or repeat encoder calls. Before any forward, check every source
has at most 64 input tokens including EOS, both fixed label sequences have at
most five tokens including EOS, and the pinned model has hidden width 512 and
vocabulary size 32,128. Preserve exact IDs and masks; a source that fails
preselection is ineligible, while an unexpected runtime dimension or evidence
bound violation stops the affected pass as incomplete. These limits allow at
most 14 retained logit rows per evaluation. Independently counted pass payloads
use `evaluations*(14*32128*4 + 2*64*512*4) + encoders*(64*512*4) +
groups*(40*512*4) + 68*1024*1024` bytes. The third term reserves 40 landing
vectors per group; the last term is a separate 64 MiB journal and 4 MiB
failure reserve per pass. With eight groups/36 encoder calls per discovery
pass and four groups/18 calls for painting, this gives 1,272,238,080 bytes
for each discovery pass and 671,770,624 bytes for painting, excluding raw Gaussian source vectors and owner, custody
and selection artifacts. Their sum is 3,216,246,784 bytes, about 2.995 GiB.
Use per-pass caps of 1.25 GiB, 1.25 GiB and 0.75 GiB respectively, including
the additional saved Gaussian vectors. Before any forward, require at least
4.5 GiB free on E: for those caps, 0.25 GiB of owner/selection artifacts and
1 GiB further headroom. This check does not reserve filesystem space. Verify
the exact frozen schedule and destination free space at launch, and stream/cap
storage accordingly.
Use a versioned journal sized for this schedule, not the earlier study's
smaller hard cap. This file-evidence cap is not a GPU memory limit. This is a causal intervention on the chosen
input-state row, conditional on specific donors, controls, doses and prompts.
Neither a positive nor negative result establishes the location or absence of
all sentiment information, or capability preservation beyond the measured
outputs.
