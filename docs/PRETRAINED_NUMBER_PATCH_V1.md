# DistilGPT2 agreement and known-injection study — protocol v1

**Research protocol, not a completed experiment.** At its initial publication,
no held-out cases have been generated or run under this protocol. This study
asks two distinct questions. First, does the model prefer
the grammatically agreeing continuation in a small, counterbalanced set of
subject–verb prefixes, and do particular donor activation replacements change
that preference? Second, can the same instrumentation locate and reverse a
deliberately injected, known change? Success on the second question is a sanity
check for the instrument, not evidence that it discovered the source of the
model's natural behavior.

The experimental unit is a four-prompt noun/context family, not an individual
logit, site, or intervention. There are four exposed discovery families and
four procedurally selected held-out families. This fixed, small study reports
descriptive results, including negative and incomplete results; it does not
estimate a population accuracy or establish a unique agreement circuit.

## Model and execution

Use `distilbert/distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, with all loader assets verified,
including `model.safetensors` SHA-256
`e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b` and
`tokenizer.json` SHA-256
`8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`.
Verify the pinned tokenizer encodes `" is"`, `" are"`, `" was"`, and `" were"`
as the single IDs 318, 389, 373, and 547 respectively. Use float32, evaluation
mode, eager attention, batch size one, no dropout, no KV cache, no generation,
no gradient, and unpadded all-one masks. Every root forward receives one
prefix, at most 32 tokens, with `add_special_tokens=False` and no truncation.
Record the exact source revision and protocol hash, asset hashes, Python,
PyTorch and Transformers versions, actual execution device and driver,
backend/numeric settings, and RNG seeds.

This protocol is not itself evidence that the GPU admission, ordered hooks, or
custody path is implemented. Before the experiment, independently test the
actual composite capture/intervention path against a serial reference for
hook ordering, same-site actions, tensor selectors, post-action capture,
failure, cleanup and storage limits. A bounded dedicated single-model CPU
executor is permitted without GPU admission. On CUDA, maintain the shared
owner/device lease for the full owner process, acquire the applicable exclusive
execution admission for intervention forwards, enforce the measured whole-batch
profile and live memory reserve, and synchronize and drain before release. If
the CUDA path is unavailable, record that fact; it need not block CPU research.
Record whether the actual study ran on CPU or GPU; CPU preflight is not GPU
certification. Fixed public development-only pilots may measure the workload,
and their inputs, purpose, environment and outputs must be recorded as exposed.
No latency or throughput claim follows from this study.

## Behavior, public preflight, and alignment

For logits `z` at the last input token, let
`m(z) = z[318] - z[389]`. The model is correct for a singular subject if
`m > 0`, and for a plural subject if `m < 0`; zero is incorrect. This measures a
fixed two-token next-continuation preference, not a generated sentence or
general language competence. The secondary contrast `z[373] - z[547]` is
reported but never used to select a site.

The **public exploratory preflight** consists of the following 16 exact
strings, with no trailing space. Each row denotes four strings of the form
`The {subject} {context}`:

| Subjects, in order | Context |
| --- | --- |
| cat, cats, dog, dogs | near the windows |
| teacher, teachers, doctor, doctors | in the room |
| artist, artists, writer, writers | beside the rivers |
| farmer, farmers, sailor, sailors | behind the house |

The initial public proposal used `baker/bakers` in the third family. A
tokenizer-only check found `bakers` takes two tokens, so this pair was replaced
before the preflight model forward. The initial proposed held-out sampling
frame likewise included `painter`, `gardener`, and `dentist`; tokenizer-only
checks showed at least one form of each split into multiple tokens, and they
were replaced before any held-out allocation. No model outcome was used for
these structural amendments. Preserve the initial preflight proposal and the
amended pre-forward manifest as separate records; do not represent the
amendment as having been present in an earlier frozen manifest.

These are intentionally exposed development and discovery material. They are
not held out. The preflight's runner, literals, token checks, model identity and
criterion must be published before its first model forward, and all per-prompt
four contrast logits, margins and token IDs must be retained. Full-vocabulary
preflight logits are optional. Before any model forward,
check that appending each of the four contrast strings leaves the entire prefix
tokenization unchanged and adds exactly the stated single ID. In each family,
all four prompts must have the same length, the same tokenized context suffix,
and a uniquely identified, single-token subject at the same position. Check
this from the tokenizer output; do not infer position from character count.
Reject empty, over-length, duplicate or structurally misaligned inputs before
running the model. Structural failure aborts that attempt; a later version may
fix the grammar with a public record, without inspecting outcome-based
alternatives.

The preflight criterion, fixed before the forwards, is at least 12/16 correct
overall, at least 5/8 for each subject number, and at least 5/8 for each of
attractor-congruent and attractor-incongruent prompts. Report the actual counts
and all margins. Below the criterion, publish the negative preflight and stop
the natural-behavior track here; the separate known-injection track may still
proceed. A later hypothesis or input family requires a new version and must
not use held-out outcomes to tune this criterion. Passing is only a reason to
proceed, not a claim of generalization.

## Split and commitments

Discovery reuses the four exposed preflight families, but all experiment
captures and donor vectors are obtained afresh on the experiment's actual
runtime. Do not silently treat earlier CPU vectors as GPU donor activations.
For each family, the two noun lemmas share one fixed context; each lemma occurs
in singular and plural. The held-out selection uses eight distinct lemmas from
the following pool, each in the listed singular/plural pair:

`chef/chefs`, `pilot/pilots`, `lawyer/lawyers`,
`student/students`, `nurse/nurses`, `musician/musicians`, `driver/drivers`,
`singer/singers`. Tokenizer-only checks found that every listed singular and
plural subject is a single token when preceded by `The `; no held-out family
was generated for that check.

The held-out plural-attractor context pool is `under the bridges`,
`outside the houses`, `by the gardens`, `inside the buildings`; the singular
pool is `under the tree`, `outside the station`, `by the river`,
`inside the kitchen`. This paragraph is a public sampling frame, not a list
of generated held-out cases or evidence of their model behavior. The custodian
uses an independently generated secret 32-byte split nonce after protocol and
runner review, the relevant discovery predictions are frozen, and capacity and
custody are ready. Generate it once from OS cryptographic randomness, independent
of observed outcomes; do not reroll it after a failed check. For each ordered item
`x` in a pool, sort by `(SHA256(domain || 0x00 || nonce || 0x00 || UTF8(x)),
UTF8(x))`, with the domain bytes `MLG-number-v1-nouns`,
`MLG-number-v1-plural-contexts`, and `MLG-number-v1-singular-contexts` for the
three respective pools. For a noun pair the item `x` is its literal
`singular/plural` string as listed above; a context item is the literal context
string above. Take all eight sorted nouns, pair adjacent nouns, and
take the first two sorted contexts from each context pool. Sort those four
selected contexts by the same rule with domain `MLG-number-v1-family-order`,
then associate them in order with the four noun pairs. Expand each pair into
four strings in the same singular/plural order as discovery. Ties use the stated
byte order. This is a reproducible pseudorandom allocation conditional on the
nonce, not a claim that the finite pool is a representative population.

After drawing the held-out nonce, the custodian tokenizes both splits using the
pinned tokenizer and verifies the same structural checks above. Known exposure
is **allowed and disclosed** for discovery; it is the purpose of that split.
Check every held-out row against the available versioned prior exposure
inventory: reject a held-out full token row that equals any prefix of a known
inventoried row. This particular known-history check does **not** reject every
longer held-out row that extends an unrelated, shorter known row. Also reject
duplicates and whole-row prefix relationships in either direction between the
particular discovery and held-out splits. Common initial tokens alone are
permitted. Neither rule establishes semantic or pretraining independence.
Any collision, failed alignment or failed check aborts the attempt; never
redraw or replace an individual case using model outcomes. Preserve the
failure. A revised sampling frame requires a separately reviewed protocol
version. This check is limited to known history: it cannot establish lack of
unknown prior use or lack of pretraining exposure.

Before each track's discovery study forwards, publish that track's reviewed
runner and source revision, the reviewed protocol, sampling frame, collision
rule, all analysis and selection functions, exact protocol/runtime identity,
fixed configuration, and discovery strings and tokens. Each track must use the
same bound source, checkpoint and runtime for its own discovery and held-out
work. The two tracks may bind different source revisions or runtimes, which are
reported separately; do not treat one track's CPU evidence as GPU evidence for
the other. Natural discovery may proceed without allocating held-out cases.
If the injection track participates, separately sample a secret 32-byte site
nonce once from OS cryptographic randomness, independent of outcomes and the
split nonce, and set hidden injection layer
`1 + (SHA256(b"MLG-number-v1-site\x00" + site_nonce)[0] >> 6)`; eligible layers
are 1–4. Publish the site commitment before any **injection-track** discovery
forward. Publish each track's complete discovery selections, measurements and
predictions before drawing its held-out cases or running any held-out forward.
When both tracks use the same held-out cases, both prediction freezes must
precede the draw. A track that has not frozen predictions when those inputs are
revealed remains incomplete; any later held-out test for it requires a new
prospective attempt with fresh cases. Thus an injection-track failure need not
suppress a completed natural track, and vice versa. Shared clean/control or
model-integrity failures invalidate every result that depends on them. Report
each track's status and the composite status, retaining all completed records
and any available bounded partial evidence.

For reproducible commitments use these exact records. `canonical(record)` is
UTF-8 encoding of JSON serialized with sorted keys, `ensure_ascii=True`,
`allow_nan=False`, and separators `(',', ':')`, without a trailing newline.
All values in these records are strings, integers, arrays or objects, with no
floats. Digests and nonces are lowercase hexadecimal strings when published.
The site record has exactly `schema='mlg-number-site-v1'`, `protocol_sha256`
(SHA-256 of these protocol bytes), `source_commit` (the full executed source
revision), `checkpoint_revision` (the pinned revision above), and integer
`layer`. The site commitment is
`SHA256(b"MLG-number-v1-site-commit\x00" + site_nonce + b"\x00" + canonical(site_record))`.

For the discovery freeze, publish each participating track's exact prediction
file and runtime-manifest file. Each prediction file includes all discovery
measurements, rankings, selected baselines and predictions for that track. Form
an aggregate record with exactly `schema='mlg-number-prediction-freezes-v1'`,
`protocol_sha256`, `checkpoint_revision`, and `tracks`. `tracks` maps the name
`natural` and/or `injection` to an object with exactly `source_commit` (its
executed revision), `runtime_sha256` (SHA-256 of the exact published runtime
manifest bytes), and `prediction_sha256` (SHA-256 of the exact published track
prediction file bytes). Publish `canonical(aggregate_record)` without a trailing
newline and SHA-256 those exact bytes. Include only tracks whose discovery is
complete; every included file and the aggregate precede the held-out draw.
Each track's recorded runtime manifest must describe the actual backend and
settings; its hash binds the record but does not attest that it was truthful.

The held-out record has exactly `schema='mlg-number-heldout-v1'`,
`protocol_sha256`, `source_commits`, `checkpoint_revision`,
`prediction_freeze_sha256` (SHA-256 of the exact aggregate bytes defined above),
`exposure_inventory_sha256`
(the checked inventory's versioned digest), and `families`. `source_commits`
is an object mapping each participating track's name (`natural` and/or
`injection`) to its exact bound source revision; it must agree with the
aggregate record. `families` is an
ordered array of four objects, each with exactly `family_index` (0–3),
`context`, `subjects` (the four ordered literal noun forms) and `cases`. Each
ordered `cases` array has four objects with exactly `text`, `input_ids`,
`attention_mask`, and `subject_index`. Inputs and masks are integer arrays.
The held-out commitment is `SHA256(b"MLG-number-v1-heldout-commit\x00" +
split_nonce + b"\x00" + canonical(heldout_record))`. Publish this commitment
and the discovery-freeze digest before the first held-out forward. At completion
reveal the split nonce and held-out record; if the injection track participated,
also reveal its site nonce and site record. Recompute each applicable commitment.
Record observed publication order and any disclosure, but a digest or local
clock is not proof of publication time or faithful execution. Because team
members share an OS account, this is procedural blinding rather than secure
isolation; errors, filenames, logs, order or telemetry must not disclose the
site to the selector. If exposed, label the attempt unblinded.

## Candidate boundaries and natural-behavior interventions

The eight prespecified candidates are the last-input-token vectors at
`transformer.h.{1..4}.attn` output tuple element 0 and
`transformer.h.{1..4}.mlp` direct tensor. They are the projected attention and
MLP outputs before their residual additions. The selected batch index is zero,
the invocation is zero within the root forward, and the local token index is
`len(input_ids)-1`. Their exact paths, selectors, shapes, dtype, source/saved
device, run ID and execution order are recorded and checked. Validate these
boundaries against the pinned architecture; do not infer semantic equivalence
from matching shape. All stored values and derived metrics must be finite;
otherwise the affected track is incomplete. Unless explicitly stated otherwise,
`logits` here means the 50,257-element vocabulary vector at the last input
position, not the full sequence output. Transient full model outputs and other
working allocations are outside the retained-payload cap and must be covered by
the separately measured resource profile.

For each target within a quartet and each site, patch four clean sources:

1. Same lemma and number: capture and self-patch in the **same invocation**.
2. Same lemma, opposite number: the other number of the target lemma.
3. Different lemma, same number: the other family lemma with the target number.
4. Different lemma and opposite number: the remaining family member.

Cross-input donors are captured from the named source's clean forward on the
same model, backend and runtime; they are immutable and associated with exact
input, site, invocation and token metadata. No matching by a bare shape. Source
and target have the same context and verified sequence length/position.
For target `t`, let `w_t = +1` for singular and `-1` for plural, and define
`u(t,s,donor) = w_t * (m(clean_t) - m(patched_t_from_donor))`. Thus a positive
value is a shift away from the target's correct-number margin. For each family
and candidate `s`, average `u` over all four targets in each donor cell, writing
the resulting cell means `E[same-lemma,same-number]`,
`E[same-lemma,opposite-number]`, `E[different-lemma,same-number]`, and
`E[different-lemma,opposite-number]`. The fixed primary contrast is

```
D(f,s) = 0.5 * (
    E[same-lemma, opposite-number] - E[same-lemma, same-number]
  + E[different-lemma, opposite-number] - E[different-lemma, same-number]
)
```

Positive `D` supports a larger target-oriented shift under these opposite-number
donors than under these same-number donors. Report every target's raw clean and
patched margins, signs and sign flips, all four cell means and `D`; do not
silently drop cases because the unpatched model was wrong. `D` is a main
contrast averaged across the two lemma conditions, not their interaction.
A positive value can arise from erasing the target margin toward zero without
transferring the donor's number preference; it does not establish transfer.
Decode stored float32 logit bit patterns to exact binary64 values before
subtraction, including subnormals. In preflight and both tracks compute each
two-logit margin with `math.fsum((z_first, -z_second))`; compute each signed
shift with `math.fsum((m_clean, -m_patch))` times `w`. Use `math.fsum` for
each four-target cell and each four-family mean, dividing by four, and compute
`D` as half the `math.fsum` of its four signed cell terms. Each mean has exactly four equally weighted
contributors; a missing observation makes that track incomplete rather than
changing its denominator. Same-number different-lemma donors are a lexical
control, not an assumed zero effect. The factorial reduces some confounding,
but a cross-prompt replacement changes more than grammatical number and may
form an unnatural hybrid state.

On discovery only, compute mean `D` across the four families for each of the
eight sites. Select the largest strictly positive mean; if none is positive,
abstain. Resolve ties in layer order, attention before MLP. Freeze the full
ranking, all effects and `p_s = mean_f D(f,s)` for every site. Also freeze:
the earliest site (layer 1 attention); a random site selected once using the
ordered list above and `SHA256(b"MLG-number-v1-random\x00" +
(20261028).to_bytes(8, "big"))[0] % 8`; and the site of greatest discovery
mean activation L2 difference between singular and plural sources of the same
lemma, ties by the same order. Count each of the two lemma pairs in each family
once, not as separate directed comparisons. Decode activation bits as above; compute each
L2 as `math.sqrt(math.fsum((a-b)**2 for each of 768 coordinates))`, then
average the eight lemma-pair norms with `math.fsum(...) / 8`. These baselines must receive the same held-out
measurements. The magnitude baseline is an association, not a causal claim.

For held-out families report all clean margins/accuracy, all site/cell effects,
the selected and baseline sites, and each site's mean and individual `D`.
Compare each frozen prediction `p_s` with each held-out family `D(f,s)` using
MAE and RMSE; also compute these errors for the fixed zero predictor. For four
family errors `q`, use `math.fsum(abs(q) for q in errors) / 4` and
`math.sqrt(math.fsum(q*q for q in errors) / 4)` respectively. The
four families, not their four target sentences or eight sites, are the units of
summary. Do not compute a significance claim or treat the four fixed families
as an independent sample from a defined population. A wrong clean prediction,
zero or negative `D`, failure to beat zero, abstention, or a baseline winning
is a reportable finding; none is a reason to discard a held-out family.

## Deliberate injection and localization

Independently of the natural donor contrast, add a fixed vector to the hidden
layer's last-token MLP output, invocation zero. Use seed `20261028` and the
same exact sign-vector rule as `PRETRAINED_CAUSAL_PROTOCOL_V1.md`:
bit `j` is the most-significant-first bit `j % 256` of
`SHA256(b"MLG-v1-direction\x00" + seed.to_bytes(8,"big") +
(j // 256).to_bytes(4,"big"))`; a one is positive. Each float32 component is
the bit pattern `0x3c93cd3a` or `0xbc93cd3a`, giving nominal L2 norm 0.5.
Do not tune this direction, strength or hidden site after seeing data.

For the first lemma's singular and plural prompts of each family, run clean
and injected paths and capture ordered attention and MLP last-token vectors
at all six layers plus the layer-5 block output and final last-token logits.
The plan uses model execution order: a candidate patch before the hidden site
acts before injection; at the hidden site inject, capture the post-injection
vector, patch, then capture the post-patch vector; at a later site inject first
and patch when that site executes. A reverse patch uses the corresponding
injected baseline vector as donor in a clean, no-injection forward. Captures
and actions at different sites must respect actual execution order.
With the same input, compare clean and injected MLP outputs elementwise using
`abs(a-b) <= 1e-5 + 1e-5 * max(abs(a),abs(b))`. A clean-repeat check at all
measured boundaries and logits must pass under the same rule. Each trace must
be complete and ordered, with exactly one matching invocation at each declared
boundary. An unexplained upstream difference or bad metadata invalidates the
case. The first eligible MLP boundary outside tolerance receives that prompt's
vote; no observed difference means no vote. The discovery candidate ranks
eligible layers by vote count, ties by lower layer, and abstains if every score
is zero. Freeze the ranking, the earliest layer, a separate fixed random layer
`1 + SHA256(b"MLG-number-v1-injection-random\x00" +
(20261028).to_bytes(8,"big"))[0] % 4`, and the greatest mean clean/injected
MLP-vector L2 difference site, ties by lower layer. Compute each norm by the
same decoded-value `math.sqrt(math.fsum((a-b)**2))` rule and average the eight
prompt norms with `math.fsum(...) / 8`. Record the true site's rank
only after reveal. A first observed difference is relative to these taps and
does not by itself establish a general causal source.

At every eligible layer and both selected prompts in each family, test
clean-to-injected replacement and the reverse injected-to-clean replacement.
For the former, define `e(s,i) = m(injected+patch_s) - m(injected)` and freeze
the mean discovery `e` for each site as its held-out prediction. Report the
individual effects, per-site MAE/RMSE versus both frozen and zero predictions,
and reverse effects without assuming they are negatives. Compute the scalar
injection prediction as `math.fsum` of the eight discovery prompt effects
divided by eight. For eight held-out errors `q`, compute MAE as
`math.fsum(abs(q) for q in errors) / 8` and RMSE as
`math.sqrt(math.fsum(q*q for q in errors) / 8)`, and
also report their two-prompt family groupings. Treat the two
prompts within each family together; do not claim eight independent units.
Also report full last-token-vocabulary restoration
`R_s = 1 - sum_i ||z_patch_s,i - z_clean,i||² /
sum_i ||z_injected,i - z_clean,i||²` for the held-out cases. Decode float32
values as above and use `math.fsum` for the squared differences over all
vocabulary coordinates and prompts. If the denominator
is at most `1e-8`, report `R_s` undefined and retain numerator and denominator.
Restoration, scalar prediction and correctly identifying a deliberately
altered site are distinct results. A no-effect downstream injection is valid
if its actual internal addition landed correctly.

## Controls, integrity, and budget

For every tolerance comparison in either track, decode the original IEEE
float32 bits of each stored element to its exact binary64 value, including
subnormals, then evaluate `abs(a-b) <= 1e-5 + 1e-5 * max(abs(a),abs(b))`
in binary64 in that order. This includes clean repeats and the upstream,
self, null, delivery and final-clean controls. Do not let float32
flush-to-zero hide a difference. This numerical tolerance is separate from
the raw-bit equality required for patch landing.

Every patch is immediately followed at the same module, invocation and token
by a capture. Its float32 raw bits must equal the immutable donor's bits,
including signed zero. Retain the actual bounded donor, pre-action and
post-action float32 vectors, together with identity metadata, so an independent
auditor can recompute landing. Retain full last-token vocabulary logits for
the clean, injection, restoration and controls that require them; the four
contrast logits suffice for other natural candidate patch forwards. For the
self and null actions, both the selected vector and
all final logits must also match the appropriate unpatched reference under the
fixed tolerance. The injection's post-add bits must equal an independently
computed elementwise float32 rounded `pre + delta`; record the realized delta
and require at least one changed element and finite values. Handle subnormal
bits explicitly when deriving reference arithmetic, and verify the runtime's
rounding/denormal behavior rather than allowing flush-to-zero to make a false
equality. Keep the hidden pre/post and action records private until reveal.
Reject an unplanned direct modification elsewhere or an action on the wrong
site; downstream activation changes from a planned action are expected.

For each target, patch the complete layer-5 block output at the last token from
the same-lemma opposite-number donor: in this no-cache architecture the positionwise final
normalization and language-model head should reproduce that donor's last-token
logits within the fixed tolerance. Separately patch the same final-block
output at the uniquely aligned **subject token**, which must leave the target's
last-token logits unchanged within tolerance: no cross-token layer remains.
The subject-token control uses that same donor. The per-target natural null
control acts at the layer-1 attention output at the last token.
These are delivery/position controls by construction, not evidence that the
final block computed agreement. In the injection track, a clean layer-0 MLP
patch is an upstream negative control, whose logits should agree with the
unpatched injected baseline. A clean final-block last-token patch is a
downstream restoration control, whose logits should agree with the clean
baseline. An injection-path null and same-run self-patch control compare
against the unpatched injected path. On control
failure retain the observations and stop, rather than presenting the case as a
successful test.

Reject preexisting unexpected hooks, nonfinite or missing observations,
unexpected invocation/order/selector/shape, wrong patch landing, model-state
drift, failed clean repeat, cancellation, OOM or trace-budget overflow. Remove
plans and check the hook registry after each forward; hash model parameters and
buffers before and after the study, and run fresh clean forwards after each
family. Cleanup equality demonstrates restoration of the instrument, not
preservation of unrelated capabilities while the intervention is active.
Save terminal failure/incomplete status, the failed phase and sanitized error,
and all prior completed forward records. Preserve any bounded partial evidence
that is actually available. The current plan API may discard uncommitted
snapshots from the failed forward; disclose that limitation and do not claim
those snapshots were retained. Never overwrite a prior attempt or convert a
partial run into a success.

When both tracks share their matching clean baselines, the planned maximum is
178 root forwards per family: 8 clean and repeat;
128 natural candidate patches; 4 natural null; 4 final-token delivery and 4
subject-token controls; 26 injection-track forwards for two target prompts
(each: one injected baseline, four clean-to-injected patches, four reverse
patches, upstream and downstream controls, null and self); and 4 final clean
checks. The first four clean forwards simultaneously capture all required donor
sites, and each injected baseline captures all its required sites. Four
discovery plus four held-out families then total at most 1,424 root forwards.
If the two tracks use separate clean, repeat and final-clean forwards, allow
12 more per family, for at most 1,520 root forwards. These bounds exclude
recorded public pilots; they are planning checks, not timing or guaranteed
resource estimates. Bound retained float32 tensor
payload for the entire study to 512 MiB and per-forward payload to 1 MiB;
count actual unique retained snapshots, logits and action evidence, and stop on
overflow rather than truncate. These are storage limits, not host-memory peak
or VRAM reservations. On CUDA the broker must measure and reserve actual device
memory separately and synchronize before releasing a lease or recording completion.

## Claims and publication

Publish the preregistered source and manifest, frozen discovery ranking and
predictions, per-case observations and controls, status of every planned
forward, revealed commitments and site, execution environment, and all
negative/failed attempts. Independent review should verify the numerical
contrasts, source/target mapping, landing and cleanup, ordering and replay where
feasible. A checksum alone does not authenticate execution.

Prior causal-mediation work has examined agreement in other architectures and
structures ([Finlayson et al., ACL 2021](https://aclanthology.org/2021.acl-long.144/));
word-form frequency can also affect agreement behavior in transformer studies
([Wei et al., EMNLP 2021](https://aclanthology.org/2021.emnlp-main.72/)).
Neither establishes the outcome for this checkpoint. An observed patch effect
is sensitivity to a specific replacement under these inputs and execution
conditions. It does not prove a unique natural number representation, isolate
the computation from lexical or other correlated changes, establish necessity
or sufficiency, or establish preservation of broader useful capabilities.
