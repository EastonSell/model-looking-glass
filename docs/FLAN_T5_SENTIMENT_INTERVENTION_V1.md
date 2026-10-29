# FLAN-T5 sentiment: prospective matched-dose intervention

## Question and scope

On locally held-out short reviews that this checkpoint already classifies correctly,
does moving its final encoder state at the adjective toward an opposite-polarity
donor move the positive-versus-negative sequence score toward the donor more
than moving the adjective toward a different same-polarity donor, or making
equally sized opposite-donor moves at the instruction or subject position?
The prediction and the finite decision rule below concern these particular
interventions. They do not assert a unique sentiment representation, a
causal mechanism for the entire behavior, or population-level generalization.

This plan was written **after** a completed 64-input development screen on the
same checkpoint. All 64 generated the assigned bare label, but that screen's
prespecified exact-character-span token-alignment diagnostic was **0/32**:
the tokenizer included a leading space. A separately computed post-screen
diagnostic accepting one leading ASCII space gave 32/32 aligned pairs. The
new rule below is adopted prospectively for new cases; it does not revise the
old result or establish that new cases will qualify. A post-screen calculation
on the development encoder states suggested a common control-matched radius
of about 11–20% of the minimum natural opposite-word distance in those groups.
That is a descriptive observation, not a prediction or selection rule for
the unseen cases. The earlier held-out translation study's primary result was
inconclusive and its revealed cases and entire screen are exposed development
material.

## Checkpoint, behavior and complete outputs

Use `google/flan-t5-small` revision
`0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`, and the same pinned seven-file
asset-manifest SHA-256
`d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083` as the
development screen. CPU float32, offline, eager attention, one intra- and
inter-op thread, deterministic algorithms, no decoder KV cache; record runtime
and the source and asset hashes. Use the same two exact instructions as the
screen:

1. `Classify the sentiment of this review as positive or negative:`
2. `Is this review positive or negative?`

Append one ASCII space followed by `The {subject} was {adjective}.` The second
instruction can be read as a yes/no question: strict bare-label accuracy
measures format as well as sentiment. Encode each input with all-one attention
mask, including its EOS. For every clean and intervened output, begin decoding
from decoder-start ID 0 with explicit all-one source and decoder masks, and
append the lowest-ID argmax of raw final-position logits until EOS ID 1 or
four new tokens. Do not use a logits processor, sampling, forced BOS/EOS,
suppression, a length penalty or an implicit generation configuration. Save
each complete decoder prefix, full raw vocabulary logits and emitted ID.
Strict correctness requires exactly one actually emitted terminating EOS,
no other generated special/unknown/pad/sentinel ID and the assigned bare
label after NFC normalization, stripping outer whitespace and case folding;
decode without dropping specials or tokenization cleanup. Wrong, extra-text,
special-containing and truncated completed outputs remain visible misses.

For both labels, separately tokenize the lexical text with
`add_special_tokens=False`, reject empty, identical or special/unknown lexical
sequences, and append exactly one EOS. Save the IDs, masks and full logits for
the whole teacher-forced sequence with decoder inputs `[0] + label_ids[:-1]`.
Define `S(y)` as `math.fsum` of the selected float32 log-softmax values,
converted to binary64. Set `m = S(positive) - S(negative)`. This is a
log-probability ratio for two complete particular sequences, not a calibrated
class probability and not the generated answer. Save full logits and ranks
using `1 + count(logit > selected_logit)`, including EOS, for both labels.
Retain first-generation-step full-vocabulary logits so that distributions,
total variation, top-one changes and shifts outside these two label sequences
can be independently reconstructed. Missing/nonfinite output or an uncertain forward
is an incomplete attempt, not a behavioral miss or negative scientific result.

## Prospective local pool and selection

An independent custodian, after this protocol is public and before tokenizing
or scoring any new candidate, fixes an ordered private pool of 12 units. Each
unit has one ordinary subject noun, four distinct proposed evaluative
adjectives `P1,N1,P2,N2`, intended `P` positive and `N` negative in the stated
review frame, and a semantic-review disposition. The two positive adjectives
need not be synonyms and the two negative adjectives need not be synonyms.
A separate person checks the natural-language polarity and wording before
tokenizer/model inspection; any ambiguous unit is rejected and recorded among
the 12 units before commitment. No rejected unit is replaced.
No pretrained forward or output may influence pool construction or its order.

All subjects and adjectives are distinct across the committed pool; each is a
lowercase ASCII word matching `[a-z]{1,24}`. Compare lexical exclusions as
complete words after Unicode NFC and case folding. Exclude variable lexical
material already present in any known prior FLAN candidate
or screen, including the entire revealed translation pool and all 64 sentiment
development rows. Check those lexical exclusions before commitment, and check
exact input / bidirectional complete-row-prefix overlap with known prior FLAN
rows only after commitment using the pinned tokenizer. For this comparison,
remove exactly one final EOS from each row, reject malformed EOS, and test
whether either remaining full ID sequence is a prefix of the other, including
equality. Apply it to every candidate row, including later rejected units.
Shared instructions, label words and other
fixed boilerplate are allowed and are not claimed novel. Bind the exact public
source inventories checked and the scope of the custodian's report; prior model
pretraining and unknown exposures remain unknown. If any new protected pool is
known at execution time, its custodian checks the fixed proposal privately.

The pool object has exactly these keys: `schema` equal to
`flan-sentiment-private-pool-v1`, `protocol_sha256`, `runner_revision`,
`asset_manifest_sha256`, `exposure_sources`, and `units`. Hash fields are
lowercase 64-hex strings; `runner_revision` is a lowercase 40-hex Git commit.
`exposure_sources` is a fixed ordered array of objects each containing exactly
`path`, `revision` and `sha256`, binding every checked public inventory or
revealed source at its revision; paths are relative and at most 256 UTF-8
bytes, revisions are lowercase 40-hex commits and there are at most 64 source
objects. The 12 `units` appear in order and each has exactly `id`, `subject`,
`adjectives`, `semantic_ok`, `semantic_reason`: IDs are `u00` through `u11`,
`adjectives` is an object with exactly `P1,N1,P2,N2` string-valued keys,
`semantic_ok` is a JSON boolean, and
`semantic_reason` is UTF-8 text of at most 256 bytes, nonempty for a rejection.
All free text is NFC. The custodian fixes the exact prior-source path, revision
and digest list after refreshing known exposure and before drawing the pool.

Serialize the complete ordered pool and semantic dispositions as UTF-8 JSON
using `json.dumps(pool, sort_keys=True, ensure_ascii=False, separators=(',',
':'), allow_nan=False)` with no trailing newline. Draw a fresh secret 32-byte
salt from `secrets.token_bytes(32)` and publish the SHA-256 of the literal
ASCII domain `FLAN-SENTIMENT-HOLDOUT-V1`, one zero byte, the salt, another
zero byte, and the serialized bytes, before any tokenizer eligibility is
inspected. An independent reader verifies the public commitment's exact
digest and protocol/source/asset identity before tokenization begins. Preserve
and later reveal the salt as lowercase hex, complete pool,
all rejected units and checks. The public commitment
establishes a recorded order conditional on these custody observations; it
is not an authenticated timestamp, proof of execution, proof of secrecy or a
claim of population sampling. The science reviewers do not see the unrevealed
pool or its model outcomes before the fixed decision rule is public, and do
not see selected case contents or outcomes until the attempt closes.

Using only the pinned fast tokenizer and no model, inspect all 12 units in
their committed order, including semantically rejected units. For each instruction a unit yields four input rows in
`P1,N1,P2,N2` order. Each row must be at most 64 tokens, have a single final
EOS, valid attended masks, and exactly one attended, non-special, non-unknown
token for the full adjective. Its offset must either equal the complete
adjective span or include exactly its one immediately preceding ASCII space;
the offset must end at the adjective's end. Require a distinct similarly
aligned single subject token. Within each four-row group the lengths, masks
and all token IDs must agree except at the common adjective index; all four
adjective IDs must be distinct. The adjective index and the subject index must
each be identical in all four rows. The instruction control is token position 0, which must be
attended, non-special and distinct from adjective and subject positions.
Record exact texts, IDs, masks, special masks, offsets and every failed check.

Select the first four semantically accepted units eligible in **both** instructions; all 12 are
screened and recorded. If fewer than four qualify, retain the incomplete
selection and do not run a pretrained forward. Freeze selected texts, rows,
donor mapping and selection receipt before opening the attempt. Canonically
serialize the entire tokenizer screen, all semantic dispositions, the fixed
selection and donor mapping, together with the first public commitment and
the same protocol/runner/asset/inventory identities. Its exact top-level keys
are `schema` (`flan-sentiment-private-selection-v1`), `pool_commitment`,
`protocol_sha256`, `runner_revision`, `asset_manifest_sha256`,
`exposure_sources`, `unit_screens`, `selected_units`, and `donor_mapping`.
The 12 `unit_screens` are ordered by unit and contain exactly `unit_id`,
`semantic_ok`, `lexical_exclusion_ok`, and `instructions`. Each of the two
ordered instruction entries contains exactly `instruction_index`, `cases`,
`source_comparisons`, `eligible`, and `failure_reasons`; `cases` records the
four `P1,N1,P2,N2` rows, their complete text, IDs, attention and special
masks, offsets and adjective/subject positions, including null positions for
failed checks. `source_comparisons` records the checked source identity,
comparison count and collision boolean without public private-row hashes.
`selected_units` contains up to four IDs in selection order. `donor_mapping`
binds each selected unit's fixed reciprocal opposite- and same-polarity maps.
Use the same UTF-8 canonical JSON serialization as the pool. The exact
case-entry and comparison-entry keys and types must also be fixed in the
published tokenizer-only selector and its tests before selecting.
Draw an independent
32-byte `secrets.token_bytes` salt and publish/independently observe the
SHA-256 of ASCII `FLAN-SENTIMENT-SELECTION-V1`, a zero byte, this salt,
another zero byte and those canonical JSON bytes before any forward. Keep
both salts and private rows secret until the fixed attempt closes; never
publish unsalted private-row hashes. If selection has fewer than four units,
publish that no-forward outcome and reveal the complete pool, both salts,
screen and failed checks. Otherwise reveal them after the attempt, including
when it fails or is incomplete. If an interrupted stage never created the
second salt/card, reveal what exists, report the absent phase, retain a
sanitized failure record and do not retry this fixed attempt. Public-card
observations, private stdout/stderr and partial custody receipts count within
the 8 MiB custody limit included in the journal/metadata allowance below; use
the separate failure reserve if that limit is reached. The selected rows remain
unscored held-out inputs conditional on the documented local custody, not a
random population sample or proof of no pretraining exposure. Reveal the whole
pool and tokenizer screen after the attempt. No output-based replacement is
allowed. The selected plan contains four lexical/subject units, two instruction
groups per unit and four cases per group: **8 groups, 32 cases**.

## Clean baseline and controls

Before any intervention, save each selected case's complete encoder output,
strict greedy result and both teacher-forced sequences. A group is
baseline-supported only when all four cases produce their assigned strict
labels, with positive `m` for P cases and negative `m` for N cases. Retain
all eight group statuses and 32 individual observations. A wrong or truncated
but completed answer is a baseline miss, not an exception. Do not alter the
fixed selection or borrow donors from another group. A unit enters the
predeclared primary numerator only if both instruction groups satisfy all
baseline and intervention eligibility requirements; an unsupported group
stays in the denominator and is reported separately from a measured failure.

The action site is the final encoder stack output supplied to decoder
cross-attention, after the encoder's final layer normalization and evaluation
dropout. At evaluation the latter is identity. Preserve the input encoder
state, copy it for every action, modify only the designated row and position,
and check bit-for-bit that every other state element is unchanged. Use the
same source mask and explicit decoder masks. Verify, for all clean cases,
that unmodified encoder-output passthrough and token-self replacement at each
of the three sites match the clean generated IDs and every full decoder logit
bit-for-bit. For each of the 32 receivers, a complete encoder-state swap with
its mapped opposite-polarity donor must reproduce that donor's full clean decoder logits
and generated IDs. Repeat each unit's `P1` case in its first-executed
instruction, including a fresh encoder call, after that unit's actions; require
full-logit and generated-ID equality. Missing or mismatched controls
make the attempt incomplete, not an ordinary baseline miss.

Within each group, use fixed reciprocal mappings: opposite polarity
`P1↔N1` and `P2↔N2`; same polarity `P1↔P2` and `N1↔N2`. For each of four
receivers and each nonzero dose, compare:

* opposite-polarity donor at the adjective;
* same-polarity donor at the adjective;
* opposite-polarity donor at instruction position 0;
* opposite-polarity donor at the subject position.

Each control is active: context states can encode the adjective, and a
same-polarity adjective is a different word. A matched norm is not matched
functional disruption. Also save an **unmatched** full natural opposite
adjective-state replacement for every receiver as a descriptive anchor. Its
dose may be substantially larger and it is not part of the matched-dose
specificity comparison. Save output, full pre/post state witnesses, donor and
receiver IDs, position, source, target and actual dose for every action.

## Dose and landing

For each eligible group and each of its 16 receiver/action combinations,
decode the saved float32 state values to binary64 and compute
`d = sqrt(math.fsum((donor[k]-own[k])**2 for k in dimensions))` in increasing
dimension order. Set group radius `q` to the minimum of those 16 distances.
Compute and retain q and natural distances for every group, including groups
with unsupported clean baselines. Nonfinite encoder values or distances make
the attempt incomplete. If finite `q/2 <= 1e-6`, the group is dose-ineligible;
retain all distances. Ineligible
or baseline-unsupported groups skip the matched-dose and natural-reference
actions, while their already-required passthrough, self and full-state-donor
controls still run. At `q/2` and `q`,
construct each proposed state component as one final IEEE binary32
round-to-nearest, ties-to-even cast of `own + (target_dose/d)*(donor-own)`.
Compute the subtraction, division, multiplication and final addition in that
written order as separate binary64 operations, without fused multiply-add;
check the binary32/binary64 round-to-nearest environment and denormal
preservation. Compute and validate all 32 proposed landings (16 actions at
each of two doses) before running any matched-dose action for that group.
Nominal interpolation fractions never exceed one; float32 landing is assessed
separately by the recorded norms and cosines. Record nominal and achieved binary64 L2 norms from
the actual saved float32 before/after bits. Require every achieved norm to be
positive and within `min(max(1e-5, 1e-5*target_dose), 0.01*target_dose)` of
its nominal value, and each control's norm within 1% of that receiver's
opposite-adjective norm at the same level. Also calculate the cosine of the
actual `(post-own)` displacement with `(donor-own)` using ordered binary64
`math.fsum` dot products and the same norms. Every cosine must be finite and
at least `0.99`; a zero norm is a failed landing, reported with a null cosine
and reason. Record all actual cosines, and reconstruct the proposed post bits
independently. If any check fails, retain proposed
witnesses and `dose-unmatchable`, skip both matched and natural-reference
actions for that group, and do not call its patching a measured negative.
Report all natural distances, q and q/natural word distance;
the development range does not establish the held-out radius. For every
measured action also report achieved dose divided by its receiver-site vector
norm, including the natural word reference, with a null and reason for zero or
nonfinite denominators; this is descriptive and is not another eligibility
gate. For each
receiver also report the binary64 cosine of its opposite- and same-polarity
adjective donor-difference vectors, using the same ordered `math.fsum` dot
product and norms; if a norm is zero, report a null and its reason. These
four active comparisons are not a comparison against every perturbation.

## Fixed primary and secondary analyses

At the larger matched dose, for a positive receiver define donor-directed
movement `w = m_clean - m_patch`; for a negative receiver define
`w = m_patch - m_clean`. Apply the same sign convention to each control,
while reporting its absolute and signed movement. For each group let `A` be
`math.fsum(w in P1,N1,P2,N2 order) / 4`. For each control kind calculate
`math.fsum(abs(control_w) in P1,N1,P2,N2 order) / 4`; let `B` be the maximum
of the three results. Use binary64 arithmetic and comparisons at each stated
boundary. A measured group meets the exploratory specificity rule if
`A >= 0.10` nat, `A - B >= 0.05` nat, and at least three of its four
opposite-adjective movements are strictly positive. These floors are chosen
before the new cases, not validated practical effect sizes or significance
thresholds. A unit passes only if both instruction groups are eligible and
meet the rule. The finite primary outcome supports this operational
prediction if at least **3 of the fixed 4 units** pass, with a complete
attempt. Otherwise the operational threshold is not met; distinguish measured
rule failures from baseline-unsupported, dose-ineligible and dose-unmatchable
units, and describe the result as inconclusive about a comparison that was not
actually measured. Report the four-unit numerator/denominator and both
per-instruction group results; ineligible units cannot enter the numerator and
are not silently removed. No statistical independence is claimed for
the repeated contexts or multiple receiver directions.

Report separately the eight groups' and 32 receivers' individual signed
changes, baseline support, actual norms, control changes and first-step
distribution changes. At `q/2`, report the same quantities and compare
within-receiver signs and magnitudes with `q` without requiring monotonicity
or claiming a causal dose curve from two levels. Record strict generated-label
changes and actual token sequences for all arms, and changes in the two
complete-sequence scores and their first-token ranks. A score effect without a
generated reversal, a reversal under an active control, an opposite-direction
or near-zero effect, an unsupported baseline and failed eligibility are all
reported explicitly. A small matched dose can yield no reversal even where
the full natural opposite-word replacement changes the answer; neither alone
decides whether a sentiment feature exists.

For the first generated-token full-vocabulary logits `z` and `z'`, compute
binary64 softmax probabilities after subtracting the respective maximum,
then `TV = 0.5 * math.fsum(abs(p[k] - p'[k]) for k in vocabulary order)`.
Retain both full logits, the resulting TV and top-one IDs. The two instruction
groups have independently calculated q values; cross-instruction comparisons
must show both doses and remain descriptive rather than attributing a
difference solely to instruction wording.

## Ordering, bounds and evidence

Fix and publish the tokenizer selector and complete selection-record schema,
runner, operator, action order, per-attempt limits,
reconstruction/audit code and exact source/asset/case commitments before any
new model forward. Label the selected units 0–3 in order. Execute instructions
0 then 1 for even units and 1 then 0 for odd units. Within a group rotate the
four receiver order `P1,N1,P2,N2` left by `(unit + instruction) % 4`. For a
receiver of fixed original index r, rotate the listed four active control kinds
left by `(unit + instruction + r) % 4`; for original action index a, run low
then high dose when `(unit + instruction + r + a) % 2 == 0`, otherwise high
then low. Run the natural reference after those eight actions. Save the actual
order; do not reorder after seeing outputs. Before matched-dose or natural
actions, perform all clean observations, then all passthrough, self and
full-state-donor controls in the fixed group/receiver order.

The worst case has 32 clean, 32 passthrough, 96 site-self, 32 full-state
donor, 256 matched-dose, 32 natural-reference and four terminal-repeat output
evaluations: **484 output evaluations**, each with at most four generation
and two teacher-forced decoder calls, plus 32 clean and four repeat encoder
calls, for at most **2,940 component forwards**. Cap each input at 64 tokens,
each label including EOS at five tokens, vocabulary at 32,128 and hidden
width at 512; reject a mismatch before a forward. A conservative payload bound
is `484*14*32128*4` bytes for saved full-vocabulary logits plus
`484*2*64*512*4` bytes for before/after encoder witnesses plus
`36*64*512*4` bytes for clean/repeat states, plus 64 MiB for journal/metadata
and a separate 4 MiB failure reserve. Enforce a **1.25 GiB** retained-evidence
cap and at least **2 GiB** free space before model loading; these are file
limits, not a process-memory or runtime guarantee. The final runner must
verify this bound and reject an insufficient allocation before a forward.
Persist starts and completed raw tensors as they occur, retain partial/failed
attempts, have a parent record direct process exit, and do not automatically
retry an uncertain attempt. Independent model-free readers replay the
complete fixed denominators, patch landing, controls, generated tokens,
teacher scores and all saved outputs. Saved-file consistency and process-exit
observations are not independent authentication of execution.

Publish the complete pool and selection, all 32 clean observations, all
measured and ineligible groups, both doses, natural references, raw evidence,
source and audit results. If execution or controls are incomplete, publish
that status and the partial record without a primary outcome. This is a small
locally held-out test conditional on selected clean behavior; wording,
lexeme-specific effects, instruction ambiguity, dependent contexts, unequal
semantic distances, non-equivalent active controls and off-manifold state
interpolations limit interpretation. Neither a positive nor negative result
identifies a unique mechanism.
