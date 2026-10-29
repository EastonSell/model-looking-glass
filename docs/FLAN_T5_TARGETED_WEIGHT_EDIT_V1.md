# FLAN-T5 single-token weight change: prospective CPU study

## Question and prior exposure

Can one fixed change to a cached model's parameters change the complete
sentiment label on new wordings containing one familiar adjective, and what
else changes relative to controls with comparable measured effects? This is a
small, non-blind, diagnostic study. It is not a claim that a useful factual or
semantic memory was edited. Its targets use new wordings, not unseen words,
concepts, subjects drawn from a population, or a previously sealed holdout.

The pinned FLAN-T5-small development screen already exposed 64 inputs and
produced the assigned strict bare label on all 64. These include both
`wonderful` and `awful` in four subjects and two instructions. The earlier
screen's prespecified exact-character alignment was 0/32; its 32/32 allowance
for one preceding space was a post-screen diagnostic. This new study fixes the
allowance prospectively and does not revise that result. The second old
instruction can also be read as a yes/no question, so a bare-label result
measures format as well as sentiment. The already observed screen is used as
discovery, not as blind evidence.

Use `google/flan-t5-small` revision
`0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` and seven-file asset manifest
SHA-256 `d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083`.
Offline CPU float32, eager attention, evaluation mode, one intra-op and one
inter-op thread, deterministic algorithms, no decoder KV cache, explicit
all-one masks, and decoder start 0/EOS 1 are fixed. The checkpoint config
specifies `tie_word_embeddings=false`: the encoder and decoder share the input
embedding parameter, while the output LM head is separate. Verify the actual
parameter identities and nonoverlapping storage before any case forward.

## Fixed intervention and controls

The discovery cases are the existing screen's indices 0 and 32 and their
`awful` counterparts 1 and 33. Separate existing indices 8 and 40 and donor
counterparts 9 and 41 are the only efficacy-calibration cases. Their exact
published texts and token rows are bound in the case file. Neither pair is a
new or blind observation. Require fresh clean greedy outputs and complete
label scores to agree with the original expected positive/negative labels;
retain all outcomes and stop the edit phase as ineligible if they do not.

The fixed primary change overwrites only row 1627 (`wonderful`) of the shared
input embedding with checkpoint row 16242 (`awful`), rounded as stored
float32. Both token IDs were established with the pinned tokenizer. Before
any forward, require each discovery/calibration pair and every later fixed
target/donor text to have exactly one attended, nonspecial adjective token,
whose offsets cover the complete adjective with either no prefix or exactly
one immediately preceding ASCII space. Require equal masks and token IDs at
all other positions, including final EOS. A failed check is recorded and makes
the fixed study ineligible before any model forward; do not replace or revise
a case. This check is a prospective rule for this study.

This primary edit is a surface-token substitution: on an eligible encoder
input its input embeddings match a clean run of the corresponding text with
`wonderful` replaced by `awful`. Compare teacher-forced full logits only at
identical decoder prefixes, and compare autoregressive logits only while the
actual prefixes agree; after divergence retain both sequences without pairing
later logits. Also compare the complete generated sequences. This equality
is expected by construction when the decoder prefix does not contain the
edited token, and is not evidence of a general sentiment representation.
The shared row also affects decoder *input* embeddings if the edited token is
fed during decoding. The separate LM-head row is not directly overwritten;
record whether either adjective is ever present in a decoder prefix. For each
compared identical prefix that does not contain ID 1627, require the full raw
float32 logits to agree bit-for-bit between the edited target and clean
substitution. On an unexplained mismatch retain both rows and classify the
attempt incomplete; do not infer that the substitution interpretation holds.
Retain all generated-prefix divergence and non-comparable later steps.

Use four fixed active control directions, always reconstructed from the
same untouched checkpoint rather than composed. A row's realized dose is the
binary64 L2 norm of its actual float32 post-minus-pre values; also retain the
original and changed row hashes and intended norm.

1. **Broader output control:** only the distinct LM-head row for token 2841
   (`negative`) is changed to `head_negative + a*(head_positive -
   head_negative)`, using checkpoint row 1465 for `head_positive`. This
   can affect the negative vocabulary logit regardless of the input tokens,
   rather than changing shared label-input rows; a zero dot product could leave
   a particular logit unchanged. It is not presumed behaviorally equivalent.
2. **Random input control:** use seed 20261029 in Python 3.11's
   `random.Random(seed).gauss(0, 1)` to draw 512 binary64 values, subtract
   their projection on `d = shared_awful - shared_wonderful`, and normalize
   the resulting vector to `||d||_2`. Change only the wonderful input row to
   `shared_wonderful + a*random_direction`. A zero or nonfinite norm makes
   the attempt ineligible. Save the resulting frozen direction and its hash.
3. **Opposite input control:** change only the wonderful row to
   `shared_wonderful - a*d`. It is an active opposite-direction control and
   may fail to have the target effect.
4. **Same-polarity input control:** replace only the wonderful row with the
   checkpoint row 1287 (`excellent`). This is a fixed full replacement and is
   an active semantic control, not a claim of matching target efficacy.

Construct directions using binary64 arithmetic on promoted checkpoint
float32 values. In fixed index order use `math.fsum` for each dot product and
sum of squares, `math.sqrt` for norms, and `math.fsum` for the two terms of
each projected coordinate. Construct each new coordinate in binary64 and
round once to float32 in an offline scratch array, then hash the candidate row
and record intended and realized norms for every grid point before any edit.
Install each point only once in its arm, verify its bytes against the frozen
candidate before the first forward, then restore and probe once. Verify
round-to-nearest-ties-to-even with two float32 midpoint probes and verify
float32 subnormal preservation before and after construction, scoring and
forwards; an unavailable or changed mode aborts with incomplete evidence.
Verify that precisely the intended row and its documented shared aliases are
changed, with the separate LM-head storage unchanged for input-row arms. A
zero/nonfinite direction detected before work makes the attempt ineligible
without an edited forward; uncertain restoration is incomplete. For controls 1--3 use
exactly the positive dose grid `a = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
2.0, 3.0, 4.0)`. The clean zero dose is measured separately and is not a
substitute for an active control. The primary and same-polarity edits use
their fixed full replacement, without grid selection.

## Efficacy selection and frozen execution order

For each complete label sequence, tokenize the literal lexical label without
special tokens and append EOS: positive `[1465,1]`, negative `[2841,1]`.
Save the teacher-forced raw logits and compute token log probabilities in
binary64 with a shifted log-normalizer. Define `M = S(negative) - S(positive)`
and `N = S(negative)`. For thresholds compute each score contrast and
changed-minus-clean effect in one `math.fsum` over promoted selected logits,
row maxima and shifted log-normalizers, rather than subtracting separately
rounded printed scores. Retain the separately displayed scores too. This
reduces cancellation; the exponentials and logarithms are still rounded, so
near-zero signs are not exact-arithmetic proofs. Actual greedy labels are a
separate observation; a margin alone is not a generated answer.

First measure clean discovery/calibration and clean paired `awful` inputs;
then the fixed primary edit on discovery/calibration; then every grid point
for each of the three grid controls on both calibration cases; then the fixed
same-polarity control on both calibration cases. No target or locality output is examined during this
selection. The primary must increase both M and N by at least 0.5 on each
calibration case and produce the exact negative label on each; otherwise the
comparison is ineligible. Nevertheless execute the fixed calibration grid and
all frozen target/locality arms descriptively when this efficacy gate alone
fails, and retain the final overall ineligible status. A failed initial clean
discovery/calibration gate stops before any edit; an uncertain forward or
restoration is incomplete and stops the attempt.

A grid point matches only if, separately for each calibration case and both
effect metrics, the absolute difference from the primary is at most
`max(0.10, 0.10*abs(primary_effect))`, both primary effects satisfy the above
positive floors, and the control's complete strict greedy label is the same
negative label as the primary's on both cases. Use the matching point that
minimizes the maximum of these four absolute-difference/tolerance ratios;
break ties by smaller dose `a`. If none match, record `match=false` and use the
same deterministic nearest positive-dose point only for a clearly labeled
unmatched descriptive evaluation, even when the primary floor fails. The
nearest point minimizes the same four normalized discrepancies and uses the
same smaller-dose tie break. A 0.10 absolute tolerance at a 0.5 effect allows
a 20% difference: this is finite approximate balance, not equal efficacy.
Never expand the grid, change a direction,
relax a tolerance, retry, or omit a failed control after inspecting outcomes.

Publish the complete discovery, all calibration grid results, selection and
identity before the first target/locality forward, together with independent
replay of every grid point and its selection and independent checks of saved
process and restoration records.
Then evaluate the fixed primary, same-polarity and three selected grid points
on every fixed target and locality, including when calibration produced no
match or clean target baselines underfill. Do not select or remove an
evaluation case based on any edited output.
Run development and evaluation as separately supervised, reaped CPU processes.
The evaluation process must independently rederive the checkpoint directions
and selected float32 rows and verify their saved development hashes, and bind
the exact published development logits, result, independent review, case file
and frozen source before its first model call. A mismatch stops the second
phase without an evaluation forward. Both processes enforce the same model,
state, numeric and restoration contract; the forward and byte caps below are
cumulative across phases.
Freeze the exact complete source for both phases and all cases, seeds, grids,
numeric rules and process/asset identities before the first discovery forward.
Any correction is an explicit new prospective protocol, not an invisible
reinterpretation.

## New target and locality inputs

Before the first forward publish exactly four semantically reviewed, distinct
new target sentences containing one `wonderful`, and their four corresponding
`awful` substitutions. Each requests only a positive/negative sentiment label.
Independent semantic review must establish that each target is positive and
its substitution is negative in context, without sarcasm, negation or
quotation ambiguity, and that each locality candidate is unambiguous.
Publish eight distinct locality cases: four sentiment examples not containing
either adjective token; two short, unrelated tasks containing `wonderful` in
different contexts; and two unrelated tasks without either adjective token.
Fix the candidate answer for each. A target-token-containing repetition task
also probes the changed shared row when that token appears in a decoder
prefix. These are fixed inputs, not a random sample; publish all cases,
expected answer strings, tokenizer IDs, semantic-review dispositions and
prefix/exposure comparison before any new case forward. Reject overlap with
known prior FLAN complete input rows in both prefix directions after removing
one final EOS, except for the eight explicitly named prior discovery and
calibration rows. No replacement after a failed check.

The independent FLAN sentiment custodian must receive the exact candidate
bundle and all proposal/screen token rows, including any rejected proposals,
before any new input is forwarded. Before its currently unformed private pool is
drawn, include those rows and their variable words in its exclusion ledger.
If an applicable pool has already been committed, an authorized, identity-bound
custodian comparison must instead establish the specified scoped non-overlap;
do not modify the commitment or inspect private rows. Fail closed if a required
comparison cannot be authorized or completed. Refresh the scope of every
relevant FLAN protected pool at each launch boundary. A scoped absence report
is not a private comparison, and known public exclusions do not prove anything
about unknown model pretraining history. Disclose all shared words and
instructions. No secret future unit is requested or inferred.

Run a clean baseline and clean substituted input for every target, and a clean
baseline for every locality. A target is baseline-eligible only if it produces
the exact positive label and its substituted text produces the exact negative
label, both terminated with EOS and uniquely top at each generated step.
Keep every baseline miss; at least three of four target pairs must qualify for
the fixed target success rule. Each locality's exact expected answer must be
fixed before forward; report baseline-correct and baseline-wrong separately,
never count a wrong baseline as evidence of preserved capability. Report
greedy text/IDs/EOS, changed output, the teacher-forced expected-string score,
and first-decoder-step full-vocabulary KL/total variation for every locality.
Expected strings are narrow exact-answer criteria and need not cover all
semantically valid alternatives. If neither the encoder input nor any decoder
prefix contains ID 1627, every shared-row edit (primary, random, opposite and
same-polarity) is structurally a no-op on that locality: unchanged output and
zero KL there are identity checks, not evidence of learned selective editing.
Report the two target-token-containing locality
cases separately. The output-row control can affect all inputs by construction;
same-row random and opposite controls are the closer localization comparison,
only if they meet the fixed efficacy requirements.

After the clean and calibration prerequisites pass, the primary behavior is
**positive** if at least three baseline-eligible
targets produce exactly negative plus EOS with a unique top at every step
under the fixed edit; otherwise it is negative, or ineligible if fewer than
three targets were baseline-eligible or an earlier gate failed. Report the
whole four-row table and all baseline/underfill failures. This says only what
happened to these fixed wordings. A matched-locality comparison is eligible
only when the control matched on calibration, the fixed target outcome is
positive, and *every* baseline-eligible target separately remains within the
same per-case M and N tolerances and produces the same strict label in the
primary and control. Otherwise report its locality values as unmatched or
ineligible, with no superiority claim. Even passing this finite balance rule
does not prove functionally equal efficacy or causally explain any KL gap.

## Evidence, restoration and resource boundaries

For every clean and edited case retain the full raw float32 vocabulary logits
at each greedy generation step and each step of both label teacher-forced
sequences, actual decoder prefixes, emitted IDs, ties, scores, case identity,
and hashes. Greedy decoding uses raw logits, lowest-ID argmax and at most four
new tokens, with no processor, sampling or forced EOS; the strict outcome also
requires a unique maximum, exactly one terminating EOS, no other special IDs,
and exact decoded text after NFC, outer-space stripping and case folding.
Other completed outputs are misses; ties and truncation are visible. The
non-sentiment locality candidate is also separately teacher forced with EOS.

Before each arm verify original state and nonpersistent-buffer digests, and
after every mutation restore the checkpoint row in `finally`, verify the
entire state and buffer digests and repeat a fixed clean full-logit probe.
The probe uses frozen discovery case 0 with decoder prefix `[0]`, records one
full-vocabulary row and counts as one model call. Installing a selected row
again in the separately supervised evaluation is a distinct arm requiring its
own restoration and probe; compare all clean probe bytes to the first probe.
Fail closed on uncertain restoration, nonfinite logits, source/asset mismatch,
incomplete or uncertain execution, missing process teardown, or unknown
current custody; retain partial evidence and do not retry the fixed attempt.
The parent-owned subprocess, its child, import/runtime identities, direct
process wait and teardown must be recorded without claiming independent
execution attestation. Independent saved-record and scientific replay follow
quiescence. Distinguish ineligible, complete negative and incomplete states.

Limit input rows to 64 attended tokens including EOS, and locality-candidate
lexical answers to three tokens plus EOS. The fixed schedule has at most 144
case/condition evaluations, each needing at most one encoder, four greedy
decoder and two label-teacher calls. The four unrelated localities can each
need one additional candidate-teacher call in the clean and five edited arms
(24 calls); 36 fixed clean probes cover the pre-edit reference, a fresh
evaluation-process reference and every restore. This is at most 1,068 counted
model calls before any declared
diagnostics. Every model/encoder/decoder/probe invocation counts, including
failures. Full vocabulary size is 32,128; at most 1,284 saved vocabulary rows
are expected under these bounds, plus bounded metadata and failure reserve.

Use the existing verified cache and checkout; do not copy model weights or
existing evidence. Cap the attempt at 1,200 model calls, 256 MiB of saved
evidence plus a 1 MiB failure reserve, and a fixed 45-minute timeout per
supervised phase. Before each launch require at least 2 GiB free beyond that
evidence budget, and report the
actual observation. If the budget or runtime boundary is reached, preserve an
incomplete attempt. This cap is an execution bound, not a scientific rule.
