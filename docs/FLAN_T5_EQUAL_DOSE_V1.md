# FLAN-T5 encoder-state matched-dose follow-up v1

This public, exploratory follow-up is frozen before any pretrained forward on
the cases below. The earlier book/house experiment observed large bidirectional
effects from replacing the final encoder word state, but its instruction-state
replacement had a roughly 3.78-times smaller L2 dose. That observation motivates
this comparison; it does not establish specificity or generalization. Cat/dog
in the earlier study did not satisfy its baseline precondition. This study uses
new public pairs and does not directly resolve the earlier book/house dose
confound.

Use the same pinned `google/flan-t5-small` revision
`0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`, asset manifest SHA-256
`d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083`,
CPU float32, one intra-op and one inter-op thread, deterministic algorithms,
eager attention, seed 20261028, no cache, offline assets and CUDA hidden.
Set `MKL_CBWR=COMPATIBLE`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` before
importing PyTorch. Record exact software, tokenizer and asset hashes and loading
diagnostics. No hidden cases, GPU service, or pretraining-independence claim is
involved.

## Fixed public cases

There are three new fixed pairs, each evaluated under both templates, in the
order listed:

| English source A / B | German target A / B |
| --- | --- |
| apple / orange | Apfel / Orange |
| water / fire | Wasser / Feuer |
| bird / fish | Vogel / Fisch |

The templates are exactly `Translate English to German: {word}` and `What is
the German word for {word}?`. The fixed context-only inputs are respectively
`Translate English to German:` and `What is the German word for`; the latter
omits the question mark and is a descriptive prior, not an identical source
context. All strings, token IDs, masks, token offsets and targets including
EOS are fixed in `FLAN_T5_EQUAL_DOSE_CASES_V1.json` before any model output.
Tokenizer-only preflight found all six cells aligned, with differing word
position 6, unchanged instruction position 0, and a second unchanged context
position 5 (`:`) in the first template or 7 (`?`) in the second. These
punctuation controls have different tokens and relative positions; report them
separately by template. Instruction position 0 is partial `▁Translat` in the
first template and `▁What` in the second. Source and target offsets and all special-token flags
must be rechecked with the pinned tokenizer; no alternative cases or positions
may be chosen after a failed preflight or observed model result.

The two source encodings within a cell must have identical masks and length,
and differ at exactly one non-special token overlapping the named source word.
The three selected positions must be distinct and non-special. A failed
alignment recorded in the fixed manifest is `alignment-ineligible` and receives
no forward. A disagreement with that manifest or the pinned tokenizer is an
integrity failure and makes the attempt incomplete. Sources are at
most 32 tokens and targets at most 8 including EOS. Before intervention score
both targets, including EOS, on each source and on its template's context-only
input. For target IDs `y`, use decoder start 0, shifted decoder IDs `[0]+y[:-1]`
and an all-ones decoder mask; sum selected float32 log-softmax values in order
with `math.fsum`. This is the unnormalized teacher-forced exact-sequence log
likelihood, not greedy generation. `Apfel` and `Orange` have three and two
target tokens including EOS respectively; other target pairs have two each.
`Orange` is also a cross-language cognate. All margins can reflect target
length, lexical form, frequency and prior preference.

For each cell let `m_A = score_A(target_A)-score_A(target_B)`, `m_B` analogously,
and `delta = m_A-m_B`. Baseline eligibility requires `m_A > 0`, `m_B < 0` and
`delta >= 1.0` natural-log units. Otherwise record `baseline-unsupported`, all
baseline/context scores, and no intervention forwards for that cell; it is not
a negative intervention result. Never replace cases or retune this condition.

## Matched-dose interventions

The candidate site is invocation 0 of `encoder.final_layer_norm` during the
complete root-model forward, with the original source attention mask. The
three locations are the fixed word, instruction and punctuation positions
above. For each site obtain clean receiver and donor vectors `u_A`, `u_B` from
the baseline encoder outputs, and compute their Euclidean distance `d_s` in
binary64 using `math.fsum` of squared coordinate differences. The common
absolute dose is `q = min(d_word, d_instruction, d_punctuation)` for both
directions. This scales each natural donor difference down, never extrapolates
beyond it, and does not use score outcomes to select the dose. A nonfinite
state or norm is an incomplete attempt. If any `d_s <= 1e-6`, record
`dose-ineligible` and do not run matched patches for that cell.

For each side and site, calculate the replacement in binary64 as
`float64(own) + (q / d_s) * (float64(donor) - float64(own))`, then cast once
to float32. Recompute its actual L2 difference from `own` in binary64. It must
be finite, strictly positive and within `min(max(1e-5, 1e-5*q), 0.01*q)`
of `q` in all six side/site
combinations. For each direction the achieved instruction and punctuation
doses must each also lie within one percent of the same-side achieved word dose. Otherwise
mark that cell `dose-unmatchable`, do not run its
matched patches, and do not count it as a causal negative. Save the original,
donor and delivered vectors, all norms, scaling fractions and exact float32
bytes. Equal absolute input-state L2 does not match intervention direction,
relative activation size, downstream projection effect or information content.
Report the clean receiver-vector norm and `q` divided by that norm separately
for each site and direction; an exactly zero receiver norm has an undefined
ratio and is recorded as null. These are descriptive, not decision criteria.
T5's bidirectional encoder means either unchanged context state can itself
contain source-word information; neither is an information-free null.

For every baseline-supported, matchable cell and each of two targets, run
both source directions for these fixed conditions: nonmutating capture repeat;
word-position null; word, instruction and punctuation self patches;
complete-state self patch; complete-state cross-source swap; natural full
word-vector swap; word-vector zero; and matched-dose word, instruction and
punctuation replacements. Reuse the same target-specific decoder input and
mask. Repeat, null and all self controls must match their own clean full logits
bit-for-bit; complete-state swaps must match the donor's full logits bit-for-bit.
For each intervention use an ordered same-forward capture → action → capture
plan at the fixed invocation. The captured pre-action selection must equal the
receiver's clean baseline bits, and the post-action selection must equal the
delivered vector bits. This establishes dose relative to the actual pre-action
value, not merely an assumed deterministic rerun. Retain both witnesses and
ordered plan metadata. For each local action the selected encoder vector must equal the intended
float32 replacement exactly and all unselected output positions must remain
bit-identical to the clean receiver. Capture-repeat and full-state controls
must match their intended encoder states bit-for-bit. Check finite logits and
states, hook cleanup and model/source/asset identity. Any failed required
control, failed write or interrupted execution makes the attempt `incomplete`,
not positive or negative. Zero and natural full swaps are descriptive controls;
they do not have the common matched dose.

## Prespecified decision and evidence

For condition `c`, let `move_A(c) = m_A - m_A(c)` and
`move_B(c) = m_B(c) - m_B`. Report the signed and absolute movement of every
condition for each side, all raw scores and the context-only margins. A
completed matchable cell is `positive` only if for **both** sides its matched
word movement is at least `0.20*delta` and exceeds the absolute movement at
each matched context site by at least `0.05*delta`. Otherwise it is `negative`
at this fixed, possibly small q. Report dose-ineligible, dose-unmatchable and
baseline-unsupported cells separately; none is silently converted to negative.
The 0.20 and 0.05 multipliers are exploratory operational choices, not
statistical significance thresholds.

For the descriptive primary aggregate, a lexical pair is positive if both
templates are positive; it is negative if both templates are completed and
matchable and at least one is negative. Other pairs are unevaluable. The
aggregate is `positive` if at least two pairs are positive, `negative` if at
least two pairs are negative, and `inconclusive` otherwise. Any failed
required control or missing evidence makes the entire attempt `incomplete`.
For example, one positive and one negative template form a negative pair:
that label means the two-template replication rule was not met, not that
neither template showed a word effect.
Report all six cells and pair/template consistency; six correlated cells are
not six independent samples, and no statistical significance or generalization
claim follows. A negative concerns this matched q and these public inputs; it
does not exclude larger or differently directed effects. Even a positive result
compares only the two named control directions at this one layer and dose: it
does not establish exclusive word-token information, lower sensitivity at all
other positions, or generalization to other words.

Save each full finite float32 logits tensor and complete encoder state as
digest-addressed raw bytes, together with exact source/decoder IDs and masks,
per-token selected log probabilities, score, site/side/condition, delivered
vector, dose and control checks. Bound execution to 324 forwards and retained
evidence to 256 MiB, including partial and failure files; abort before an
over-limit forward or write and retain earlier evidence. Reserve room for one
maximum bounded return and failure metadata before each forward. A returned
component that is not a bounded CPU float32 tensor is explicitly marked
unavailable and is not materialized; bounded malformed or nonfinite returns
are retained before later validation. Use a fresh output directory,
record peak process memory, and independently audit saved bytes and score
reconstruction. The evidence limit is not a host-memory or GPU limit. Freeze
the source, case manifest and protocol publicly before execution. An independent
observer must retain the official public commit identity and check that its
source, protocol and case blobs match the runnable checkout before any model
forward. A clean local commit alone does not establish public preregistration.
The runnable accepts the observer's bounded JSON with schema
`mlg-flant5-public-observation-v1`, exact `commit`, `commit_url`, matching
`source_sha256` mapping, `observer` and `observed_utc`; it copies those bytes
into the attempt and binds their SHA-256 in `started.json`. This record is an
observer attestation, not a trusted clock or proof of model-execution chronology.
Launch a fresh interpreter with `-B -X pycache_prefix=<fresh empty path>` so
existing cached project bytecode is not used; trust in the checkout, Python and
native dependencies remains necessary. The runner enables and checks float32
subnormal handling and records the CPU numeric environment.
Run CPU-first without querying any hidden split or relying on an unrelated GPU
service. Independent score reconstruction must declare its arithmetic and a
justified tolerance when comparing float64 log-sum-exp to the saved float32
log-softmax values; the two implementations are not bit-identical.
