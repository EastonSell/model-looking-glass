# Answer-format feasibility: prospective CPU protocol

This small study asks whether a correct demonstration changes the pinned
DistilGPT2 model's full-vocabulary next-token predictions in otherwise matched
capital-city answer slots. The previous full-top screen is discovery history:
all four discovery rows topped Madrid, but their shared example was not tested
against a matched alternative. Those 36 inputs and outcomes are not held-out
evidence for this study. This protocol and its exact source, cases and
manifest must be published and independently reviewed before any new forward.

## Frozen design

Use four already-public facts in this order: France–Paris, Japan–Tokyo,
Italy–Rome and United Kingdom–London. These are four facts, not independent
samples of each wording. Freeze exactly two answer-format families, in order:
statement and question. Each family has one development template and one
different held-out template, fixed before the first forward. Each template
must ask for the country's national-capital city, end immediately before its
one-token city continuation, and be distinct from every earlier exposed input.
Independently review each held-out template as meaningfully different in
wording or syntax from its development counterpart, not just punctuation or
whitespace.
The held-out unit is a template for the same facts, not a held-out fact; the
public templates and facts are not blind or representative of all wording.

Cross each of the four facts and four templates with these four fixed prefixes,
in the order below. The string after each prefix is exactly the same target
template for that fact and role, including whitespace:

1. none: the empty string.
2. neutral: "A short geography prompt appears below.\n".
3. spain: "The capital of Spain is Madrid.\n".
4. portugal: "The capital of Portugal is Lisbon.\n".

The last three prefixes each have eight tokens under the pinned tokenizer;
verify that their concatenation preserves both the exact prefix token IDs and
the same target-suffix IDs. This controls prefix token position in those
three comparisons, not semantic content. The no-prefix condition is eight
tokens shorter and its position/length contrast must be described as such.
Portugal–Lisbon is one fixed alternative example, not a sample of all possible
demonstrations. Neither demonstration answer is any target factual city or
its fixed counterfactual continuation. Verify independently that the prefixes
state true facts, that all answer slots are unambiguous, and that the factual,
counterfactual, " Madrid" and " Lisbon" continuations each append their exact
single token without changing the prompt encoding. These continuation encodings
are metadata; only the 64 complete prompt rows are model inputs.

The fixed counterfactual continuations, for reporting only, are London for
France, Paris for Japan, London for Italy and Paris for the United Kingdom.
No edit or edit-target selection is authorized by this feasibility study.

Publish all 64 ordered case IDs, roles, family, fact, prefix, full prompt,
token IDs and continuation boundaries. Check all actual input rows for
equality and full-row-prefix relations in both directions against each other,
the latest reviewed public exposure inventory, and relevant protected scopes.
Reserve the 64 exact prompts before their first forward. Exempt only the exact
identity-bound own reservations or this attempt's authorized observations on
later checks; independently named duplicates and all other collisions remain
failures. Preserve prior and successor input rows and do not treat proposed
continuations as additional forwards.

For a relevant protected split, use either a scoped exact private comparison
or an independently checked complete public input superset whose source-bound
preparation and unchanged-split coverage are confirmed by its custodian.
Report the latter as a conditional public-support inference, not a private
comparison. A scoped report that no protected rows exist is distinct from a
pass. Check the known original natural 128-row support and the fixed public
256-row successor pool; require fresh custody at each phase. Unknown relevant
scope, collision, unverified preparation or unexplained relevant new exposure
stops the phase. No check establishes pretraining or unknown-history
disjointness.

## Phases and selection

Before any forward, freeze and review the complete executable source for
both phases, all cases, runtime, seeds, calculations, eligibility and terminal
rules. No scientific criterion, prompt, seed, threshold or executor may be
changed in response to development observations without publishing a separately
identified prospective study.

Run the 32 development inputs first. Apply Python 3.11
random.Random(20261031).shuffle(canonical_list) once to their canonical
fact-major, family-major, prefix-major order and save the exact permutation;
the seed affects execution order only, not
selection. Save every raw full-vocabulary float32 row and an ordered
case/offset/hash journal. Compute results from those saved rows. For a
development family, a fact is robust exactly when its factual token is the
unique top in all four prefixes. A family is eligible if at least three of the
four facts are robust. Select the first eligible family in the fixed family
order; save every fact's four outcomes, the full eligibility table and the
selected family (or null) and independently review/publish these bytes before
any held-out forward. This choice is baseline-conditioned and is not an
edit-target selection.

Then run all 32 fixed held-out inputs, even if no family qualified, using
random.Random(20261101).shuffle(canonical_list) on the analogous canonical
order.
Do not replace incorrect or underfilled cases, select another family, modify
prompts, or tune thresholds from held-out results. If a family was selected,
it passes this exploratory format check only if at least three of its
development-robust facts are also robust on its held-out template. A complete
selected family that does not pass is negative; no eligible development family
is ineligible, with all held-out results retained as descriptive evidence.
These labels concern only the fixed robust-across-four-prefix criterion for
these facts and templates; they do not rule out other formats or edits. If
required rows or cleanup are uncertain, the study is invalid/incomplete,
which takes precedence over both labels.

## Measurements and interpretation

From every finite saved row, promote each float32 logit to float64 and compute
shifted log-softmax. Report unique top ID and decoded token, top gap, factual,
counterfactual, Madrid and Lisbon logits and log probabilities, and factual
top status. A tied maximum has null unique-top ID, a tie count, and the
smallest tied token ID as its representative; a tie never counts as factual
or example-answer top. Retain every row, including factual errors and ties.

Within every fixed fact/template, report top changes, factual and Madrid/Lisbon
log-probability changes, KL(reference || comparison), and total variation for
all six ordered reference-to-comparison pairs: none→neutral, none→spain,
none→portugal, neutral→spain, neutral→portugal and spain→portugal. In
particular predeclare the example-specific contrast:

D = [log P(Madrid | spain) - log P(Madrid | portugal)]
  - [log P(Lisbon | spain) - log P(Lisbon | portugal)].

Report D for every fact/template and each fact's mean over its two held-out
templates, one from each family, regardless of the selected family. Compute
this contrast descriptively even when no family is eligible. The prespecified
descriptive direction is met if at least three of four fact means are strictly
positive; ties and zeros do not count. Also
report how often Spain's unique top is Madrid and Portugal's unique top is
Lisbon, and the paired changes in the factual top. Show every observation,
including opposite-direction rows. This is a finite, exploratory diagnostic:
no significance or population-generalization claim follows from four facts
and correlated templates. Changing the example changes both country and city
text, so even a directional contrast establishes only a response to these
fixed prompt contexts, not a copying mechanism. Neither unique top nor any
pairwise margin establishes a complete generated answer or stored knowledge.

## Execution and evidence

Use the pinned distilbert/distilgpt2 revision
2290a62682d06624634c1f46a6ad5be0f47f38aa, CPU eager float32, one intra-op
and inter-op thread, offline assets and the preceding full-top study's model,
site-weight, supplemental buffer, runtime and denormal checks. Each phase
uses an isolated supervised child in a trusted quiescent checkout/cache;
record source, inputs, raw rows, finite and denormal probes, state restoration,
checkpoint/asset hashes, child/supervisor IDs, model disposal, reaping and
job drain. A failed forward, nonfinite value, uncertain state/restoration,
unverified source or custody, or uncertain cleanup makes the phase invalid or
incomplete, never a numeric negative. Do not automatically repeat an
uncertain model attempt. Publish the complete evidence and an independent
model-free raw-output/selection audit. Process, custody and time records are
cooperative observations, not authenticated execution or clock evidence.
