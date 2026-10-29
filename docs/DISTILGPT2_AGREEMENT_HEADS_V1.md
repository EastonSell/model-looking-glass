# DistilGPT-2 agreement: head-slice comparison v1

**Status: prospective.** This protocol specifies a development-only head
selection followed by a separate fixed public evaluation. Until the selection,
case reservation, source and custody checks are complete, no evaluation input
is authorized for a model forward. The development inputs are already exposed;
the evaluation frames are new relative to the declared project inventory, not
secret, blind, or known to be absent from model pretraining.

## Question and prior evidence

The public natural-agreement discovery selected the layer 3 attention **output**
at the last token; opposite-number donor replacements produced a larger mean
target-oriented `is - are` change than same-number donors on its four exposed
families. Later direction and subject-form studies found substantial response
also in subjectless fragments. Those observations do not identify individual
heads or prove that an abstract number representation drives the behavior.

This study intervenes at the **input** to `transformer.h.3.attn.c_proj`, before
that projection mixes twelve 64-coordinate head outputs. It tests whether a
development-selected slice has a donor-number contrast on distinct sentence
frames, relative to a second selected head and a same-number donor change.
Head inputs, attention probabilities and post-projection contributions are
different objects; these experiments do not establish where the head attended,
an isolated circuit, or an effect in unmodified model execution.

Use the pinned `distilbert/distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, eager float32 CPU, one thread,
eval mode, uncached batch-one forwards and the repository's bounded head-slice
witness. ` is`, ` are`, ` was`, ` were` are IDs 318, 389, 373, 547. No
input-plus-answer sequence is forwarded. The last-position complete vocabulary
logits are retained; the other positions' complete logits are checked live for
finiteness but not saved for replay.

## Development selection, frozen before evaluation

Use only the 16 already-public discovery rows, in four fixed quartets:

* `The {cat,cats,dog,dogs} near the windows`;
* `The {teacher,teachers,doctor,doctors} in the room`;
* `The {artist,artists,writer,writers} beside the rivers`;
* `The {farmer,farmers,sailor,sailors} behind the house`.

Within each quartet the first two rows are targets, the corresponding
singular/plural rows of the other lemma are same-number donors, and the other
number of the same lemma is the opposite-number donor. This gives eight
development targets. Capture each row's complete L3 projection input once at
the final input position; the full input provides all 12 head slices. In a
fixed order calculate every target-to-opposite and target-to-same L2 distance
for all 12 heads, decoding float32 bits into binary64. Set a single common
`q_dev` to half the minimum of those 192 distances. If any distance is
nonfinite or `q_dev < 2**-20`, stop the scan as incomplete. Do not change the
pool, sites, dose or threshold based on outputs.

For each head and target, move its 64-vector toward each donor by the fraction
`q_dev / distance`, computing each coordinate in binary64 in the written order
`target + fraction * (donor - target)` and rounding directly to binary32
nearest-even. Require the realized L2 to match q within 1e-5 relative, the
submitted bits to land exactly and all other projection-input bits to remain
unchanged. For `m = logit(is) - logit(are)` and target sign `s = +1` for
singular and `-1` for plural, define the per-target contrast
`D = s * (m_same - m_opposite)`. It is the difference between two equal-input-
dose interventions, not a pure number manipulation: their lexical content,
direction and downstream gain can differ.

Rank all 12 heads by the mean of their eight D values, highest first; ties use
the lower head index. Freeze the winner as the candidate and the second-ranked
head as the active comparator. Keep all head/target contrasts, even if the
winner is negative; the ranking is development selection, not validation.
The runner-up is adaptively selected on the same development inputs; it is not
an independent random-head control or a randomization distribution.
Each head also gets a self-patch on the first target; observe/no-op and repeat
every one of the 16 rows. Require complete last-logit and full projection-input
bit equality for applicable no-op, self and repeat controls. The fixed scan
has 16 observations, 192 donor patches, 12 self patches and 16 repeats: 236
forwards. Save every successful forward and publish a reviewed saved-evidence
selection before forwarding the distinct evaluation frames.

## Evaluation case construction and admission

After the development selection is frozen, use the pinned tokenizer only to
select the first four eligible pairs from this ordered list: `(pilot,pilots)`,
`(guard,guards)`, `(nurse,nurses)`, `(clerk,clerks)`, `(driver,drivers)`,
`(actor,actors)`, `(painter,painters)`, `(visitor,visitors)`. Each leading-space
form must encode as one distinct token, neither form may be an exposed
development subject or the pending noun-substitution study's subject, and
selected forms/tokens must be distinct. If fewer than four pairs qualify,
stop rather than choosing an unlisted alternative. Let the first four be A–D.

Construct four frames, assigning pairs as indicated and producing singular,
plural, other-lemma singular and other-lemma plural in that order:

1. `During the storm, the {subject} beside the fences` (A, B);
2. `After the concert, the {subject} near the tower` (C, D);
3. `Before the meeting, the {subject} by the gardens` (A, C);
4. `Throughout the afternoon, the {subject} behind the barn` (B, D).

Each row must be nonempty and no more than 12 tokens, and each quartet must
have equal length and differ only at one explicitly recorded subject-token
position. Check tokenizer round-trip for ` is`, ` are`, ` was`, ` were`: each
must append exactly its one pinned ID. The 16 rows and the declared candidate
extensions are checked in both complete-row-prefix directions against the
latest reviewed public exposure inventory, the pending 54-row noun-substitution
bundle and its eight extensions, the original 128-row natural public support,
and the frozen 256-row successor pool. Exclude the intentional relation between
an input and its own extensions when comparing within this bundle. Obtain the
natural custodian's scoped report and injection custodian's exact-head/site
report, publish the fixed cases and an append-only planned reservation, then
refresh those observations and the inventory immediately before the one-shot
launch. A conflict or unavailable required check stops preparation; do not
adapt to a protected comparison. These checks cannot certify unknown prior
exposure or model pretraining.

## Evaluation and fixed interpretation

Capture all 16 evaluation inputs first. For every target in every quartet,
use the other number of the same lemma and the same number of the other lemma
as donors. Compute one `q_eval` as half the minimum target-to-donor distance
across all 16 targets, both donor kinds and both frozen heads. The dose is
determined by clean activations, without using any outcome logits. Require the
same nonzero floor, binary32 delivery and relative-L2 checks as development;
if they fail, retain the incomplete attempt and do not replace cases or relax
the protocol. This is matched **input** L2, not matched projected or behavioral
dose. Run self, opposite and same-donor patches for each target and each head,
plus clean observations and repeats: 128 forwards total.

Prespecify baseline adequacy as at least 12/16 correct clean `is - are` signs
and at least 3/4 in each frame, with ties incorrect. Report all 16 cases even
if this fails, and label the comparison baseline-limited rather than filtering
cases. Otherwise, the exploratory candidate rule passes only if mean D for the
candidate is at least 0.05 logit and strictly exceeds the mean of the 16
individual `abs(D)` values of the second-ranked head; at least 10/16 candidate
D values are strictly positive; and at least three frame means are strictly
positive and exceed the corresponding mean individual absolute comparator D.
Report both heads, every target, zero and negative values, the realized dose,
raw margins, frame means, and `was - were` as a secondary margin. The threshold
is descriptive rather than a validated effect size or significance test.
Frames and subject pairs are reused, so observations are dependent. Failure of
the rule is a retained negative result, not proof that no head or number signal
matters. A pass can still reflect lexical identity, morphology, context, model
nonlinearity or another difference between donor vectors.

## Evidence and execution limits

Run a single supervised CPU child per phase; preserve a durable call-start
record before each forward and its complete saved result after it. Record the
full projection input before/after, projection output, submitted replacement,
full last-token logits, exact input IDs/mask, coordinates, requested/realized
doses, runtime and verified checkpoint/source identities. Save clean model
state hashes before and after. Keep failed/incomplete attempts and do not
automatically retry after an uncertain forward. A 128 MiB retained-tensor
budget and separate bounded event log apply per phase; check free disk before
starting and account separately for any copies. These limits do not bound peak
RAM, allocator overhead, native work, process trees or all filesystem usage.

An independent saved-byte replay checks schedules, selection, every delivered
slice and untouched bit, full last-logit controls, dose and metric arithmetic,
and the fixed evaluation rule. A saved journal and observed child exit are
consistency evidence rather than authenticated execution or descendant-drain
proof. If interrupted during a hook or cleanup, retain the evidence and discard
the worker/model. No GPU use or timing claim is part of this protocol.
