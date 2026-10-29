# FLAN-T5 encoder-state intervention: public feasibility protocol v1

At protocol freeze, **no pretrained checkpoint has been forwarded on these
study cases**. Freeze the
completed protocol, asset identities, tokenizer IDs, code and seed before the
first forward. This small public study checks a particular intervention on an
encoder–decoder model; it is not a claim to have explained translation or
established generalization.

## Model and cases

Use `google/flan-t5-small` at immutable revision
`0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`, Apache-2.0. The official Hub
metadata identifies `model.safetensors` as 307,867,048 bytes with SHA-256
`495fa51e204676f1a857a9fc13c4c89f3f5ba9f480b898cebca02add25e6d749`.
Config, tokenizer and weights must come from this same revision; record every
asset's verified size and SHA-256. Load locally/offline, using safetensors and
`trust_remote_code=False`, and reject loading diagnostics that indicate
missing, unexpected, mismatched, or uninitialized weights. Record the exact
Python, PyTorch, Transformers and tokenizer versions.

The public development pair is `Translate English to German: cat` and
`Translate English to German: dog`, with target strings `Katze` and `Hund`.
The separately reserved exploratory evaluation pair is `Translate English to
German: book` and `Translate English to German: house`, with targets `Buch`
and `Haus`. These are public, small, and may have appeared in pretraining; the
evaluation pair is not blinded or pretraining-independent. Do not tune the
site, cases, scoring, thresholds or controls after inspecting it. A single
reserved pair cannot establish generalization.

The fixed context-only comparison is `Translate English to German:`. Score
each of the four targets under that same context and report these margins
beside source-conditioned margins. This comparison helps expose target and
context preferences; it is descriptive, and it does not remove differences
in target length, frequency or pretraining exposure. No shared Flan-T5
held-out registry was identified; protected overlap outside the separate
DistilGPT2 number study is unknown.

Before a forward, tokenize all four source strings, the fixed context-only
string, and all target strings
using the pinned tokenizer. Record exact token IDs, special tokens, masks,
source-word/instruction positions, character offsets, and target lengths,
including EOS. SentencePiece word offsets may include preceding whitespace;
validate overlap with the exact word and that remaining offset characters are
whitespace. The pair is eligible
only if the two source encodings have identical length and attention masks,
and differ at exactly one non-special token corresponding to the named source
word; position 0 must be an unchanged non-special instruction token distinct
from that word. If a pair fails this test, mark it ineligible and do not choose
replacement strings or a new position after looking at model outputs. An
ineligible pair is retained as such and receives no model forward, including
no context-only target forwards for that pair; the other fixed pair remains
eligible for its own test. Instruction position 0 is the token `▁Translat`,
only part of the written word “Translate”.

## Scoring and intervention

Run batch size 1 on CPU, float32, `eval`, no gradients or cache, one PyTorch
intra-op and inter-op thread, deterministic algorithms, and fixed seed
20261028. Require model config `decoder_start_token_id = 0`, tokenizer pad ID
`0`, and both model and tokenizer EOS IDs `1`. Score each **entire exact
target sequence including EOS** separately:
shift its labels right using the decoder start token, supply an explicit
all-one decoder attention mask for the shifted sequence, compute a float32
log-softmax over the vocabulary at every target position, convert each selected
float32 log probability to a Python float, and combine positions in order with
`math.fsum`. Record target IDs, shifted decoder IDs, masks,
per-position selected log probabilities and summed scores. This is
teacher-forced sequence likelihood, not greedy decoding; target lengths can
differ, so the score is not length-normalized and can reflect target length
and prior frequency. Report each context-only margin and its source-conditioned
difference separately without changing the primary criterion.

For source A and source B define `margin = log P(target_A) - log P(target_B)`.
Record both baseline margins before judging any intervention. The directional
baseline precondition is `margin_A > 0`, `margin_B < 0`, and
`delta = margin_A - margin_B >= 1.0` natural-log units. If it fails, retain
the result and label that pair's causal hypothesis unsupported by the planned
behavioral baseline; skip that pair's patch forwards, do not choose new inputs
or reinterpret a ratio, and proceed to the other frozen pair unchanged.

The prespecified candidate is the final encoder output **at the differing
source-word token**, after the encoder's final normalization. This is a
512-dimensional encoder-state position, not a decoder head or a 384-wide
attention-head projection. Use the repository's `ActivationCapture` and
`ActivationIntervention` sessions on `encoder.final_layer_norm` during the
specified complete root-model forwards; precomputed `encoder_outputs` alone do not test
this integration. For each pair and each fixed target, evaluate: hook-free
unmodified A and B; repeated A and B under nonmutating full-state capture;
a token-position null hook on A and B;
a token-position self patch on A and B; a complete-state self patch on A and
B; all of A replaced with B's complete encoder output and vice versa; A with its
word-position vector replaced by B's vector and vice versa; A and B with the
word-position vector zeroed; and the corresponding cross-source replacement
at unchanged instruction position 0. Always retain the source attention mask.
Run each score under the same target-specific decoder input and mask.

Require finite float32 full logits and captured encoder states. Repeat,
token-null, token-self and full-state-self logits must agree bit-for-bit with
their own baselines; complete-state swaps must agree bit-for-bit with the
donor's logits under the same target input. Compare the actual float32 bit
patterns (zero tolerance) and save the results and reconstructible raw logits.
For every patch record exact source/destination positions, special-token flags,
and the actual source-to-destination vector L2 distance. A word or instruction
patch must replace exactly the selected vector with its donor and leave every
other position bit-identical to the destination baseline. Captured-repeat
encoder states must also match their hook-free references bit-for-bit. The instruction
control is not dose matched: compare its measured distance and effect and do
not interpret a difference as proof of uniquely word-local specificity. If any
required control or source/mask/token identity check fails, the attempt is
incomplete rather than an affirmative causal result.

For the word-position patch define `move_A = margin_A - margin(A with B word)`
and `move_B = margin(B with A word) - margin_B`; define analogous signed
movements for the instruction-position control. Conditional on the baseline
precondition and all controls, the prespecified positive result requires both
word-position movements to be at least `0.20 * delta`, and each to exceed the
corresponding instruction-position movement by at least `0.05 * delta`.
Otherwise record a negative result, including all raw margins and controls.
Zero-patch and complete-swap scores are descriptive controls, not evidence
of specificity on their own. The same criterion applies to the reserved pair
without retuning.

T5's encoder is bidirectional: even an unchanged instruction-token state can
contain information about the source word. A successful patch establishes an
effect of replacing a particular final encoder state under these inputs and
controls; it does not identify an isolated source-word computation or prove
that the model used the same mechanism on other examples.

## Bounded evidence and review

Before execution save the frozen case/token manifest, checkpoint/asset
hashes, protocol and source commit, environment, and seed. Save each target's
per-position selected log probabilities and sums, each condition's margins,
identity/full-swap/untouched-position checks, patch metadata, and explicit
failure/ineligibility status. Save finite full float32 logits and encoder
states as explicitly typed, shaped raw bytes with digests, so independent
review can reconstruct the controls and scoring. Bound each source to 32
tokens, each target including EOS to 8, vocabulary to the pinned config, and
all forwards to at most 76 (four context-only plus 36 per eligible pair).
Admit writes against a 128 MiB aggregate retained-evidence cap, including
partial/failure files; abort before a write that would exceed it, and preserve
existing evidence. Baseline root forwards return the donor states needed for
the later interventions within the stated forward count. Use fresh output
directories and record actual process peak working-set memory; the retained
evidence cap is not a process-memory bound. A separate reviewer must check token alignment,
teacher forcing, masks, direction/sign, baseline eligibility and controls.
Run the development pair first; freeze its protocol and decision before
executing the reserved pair. Preserve both negative and positive outcomes.
