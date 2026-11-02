# DistilBERT agreement across neutral continuations, protocol v1

**Prospective, adaptive follow-up.** No pretrained forward on these cases has
occurred at this protocol's freeze. The preceding
[32-pair `ready.` study](../results/bert-agreement-suffix-cpu-v1-20261029/README.md)
found a large mask-probability change but limited agreement reversals. Its
single suffix, punctuation change and repeated families do not establish a
general context effect. This experiment tests a fixed small set of further
continuations on new local research cases. Their choice is informed by the
previous result; they are held out from those prior local forwards, not from
model pretraining and not a random sample of language.

## Inputs and hypothesis

The fixed [case manifest](../experiments/bert_multisuffix_agreement_cases_v1.json)
has SHA-256 `bc8d1af844c52f46bf068c04257280ed37e4302028885540ea498103d6ce125e`.
It crosses 2 surface frames, 4 subject lemmas, 2 subject numbers and 2
attractor numbers: 32 bases, with 5 paired context variants each. The subject
lemmas are rider, hunter, scholar and runner; the frame attractors are guest
and neighbor. The frames are `At the park, the SUBJECT near the ATTRACTOR`
and `Beyond the village, the SUBJECT beside the ATTRACTOR`. These are new
surface frames relative to the named checked cases, but they reuse a known
subject/prepositional-attractor syntax. All noun forms and suffixes are single
WordPieces under the pinned tokenizer. Each variant shares the identical
prefix through `[MASK]`:

1. terminal: `[MASK] [SEP]`;
2. punctuation: `[MASK] . [SEP]`;
3. present: `[MASK] present . [SEP]`;
4. outside: `[MASK] outside . [SEP]`;
5. awake: `[MASK] awake . [SEP]`.

All three suffixes are compatible with both `is` and `are`. The punctuation
variant controls for adding a period and moving `[SEP]`; a suffix-versus-period
comparison still changes the token sequence and available right context, and
does not isolate a unique internal route. Score the full-vocabulary probability
of `is` plus `are` at the mask. For **each** suffix, the prospectively fixed descriptive criterion against
punctuation alone is a positive mean of the signed paired probability differences
`P_suffix(is or are) − P_period(is or are)`, a positive difference in at
least 24/32 base cases, and a positive group-mean difference in at least 6/8
frame-by-subject-lemma groups (each group contains four subject/attractor-number
combinations). These are fixed-set rules, not
independent significance tests; report all cases and a failure for any suffix
that misses any part. Compare terminal versus punctuation and each suffix
versus terminal as secondary descriptive outcomes. Retain candidate individual
probabilities, ratios with undefined/overflow handling, `is`/`are` margin,
strict binary subject accuracy and ties, full-vocabulary top token/ties, and
per-case and grouped changes. Mark any variant with fewer than 24/32 correct
binary signs baseline-limited, while reporting its full 32 cases.

## Interventions

Use `distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, float32, eager attention,
eval/inference mode, batch size one and seed 20261029. Capture each case's
subject token at zero-indexed block-3 output, and score mask-position logits.
The donor frame, attractor number and context variant match the target. Unlike
the preceding study, **both** same-number and opposite-number donor variants
use the same other subject lemma: rider ↔ hunter and scholar ↔ runner. Thus
the donor pair varies inflection while holding its lemma fixed; singular and
plural remain different lexical tokens, and this is not proof of an abstract
number representation.

For each base pair fix one dose `q = min(2.0, all ten natural donor distances)`
across the five contexts and the two donor directions. Capture all 160 clean
variants first. If a state or distance is nonfinite,
stop the attempt as incomplete. A finite zero q or a finite rounded landing
that misses the fixed norm/vector tolerance makes the whole base dose-ineligible;
retain its clean outcomes, record the reason and continue the other bases.
Plan and validate all active landings for a base before its first intervention,
without amplifying a donor displacement. For each eligible base and context
run opposite donor, same-number donor, a seeded random direction shared across
variants, self and clean-after controls. Draw this direction with NumPy PCG64
seeded by `20261029 + case.index`. Construct intended vectors in float64,
round additions once to float32, save before/after subject witnesses, and
require finite nonzero realized displacements and norm/vector error at most
relative `1e-4`. Nonfinite landed vectors stop the attempt as incomplete.
Check clean/self/clean-after selected logits bitwise on the chosen device.

For each nonterminal variant additionally apply the **terminal-derived
opposite donor direction** at the same common q. Compare its donorward effect
with the terminal arm and the analogous shared-random context delta. This is
a descriptive fixed-intended-vector comparison: bases and contexts differ,
float32 landing may differ, and the vector can be off-manifold. It does not
isolate a downstream gain. Also record all ten donor distances and cross-context
same/opposite direction cosines; an undefined cosine remains undefined.

For each context separately, the fixed donor criterion is positive mean
opposite-minus-same and opposite-minus-random donorward signed margin, and
positive contrasts against both in at least 24/32 bases. If any base is
dose-ineligible, mark the full-set donor criterion not evaluable; retain the
valid-case contrasts as descriptive without replacing the denominator.
Donorward signed change is minus the `is`−`are` logit-margin change for singular subjects and
plus it for plural subjects. Report both contrasts, clean and intervened binary
sign flips, full-vocabulary top-token changes, and TV/KL for each intervention.
All 32 bases remain in the primary report; variant-specific correct subsets
and their intersection are descriptive only and exclude ties. Report the
8 dependent frame/lemma groups as well as individual cases. Effects of
context-specific donor directions include direction rotation; input L2 alone
does not match behavioral potency.

The maximum schedule is **1,088 forwards**: 160 clean, up to 800 ordinary
active/control arms, and up to 128 terminal-direction arms. Each ineligible
base omits its 29 post-clean arms. Full-vocabulary logits plus two
768-float subject witnesses per forward total 139,516,416 bytes; allow at most
192 MiB of retained attempt files, including journals and failure markers.
The raw float32 accounting excludes journal overhead; measure actual serialized
size in a fabricated full-schedule fixture before any pretrained forward and
confirm it fits the cap. This is an evidence-storage budget, not a GPU VRAM
limit. Preserve start and
returned records and any partial failure, classify incomplete attempts honestly,
and perform no silent retry after an uncertain forward. Use fresh relevant
host, disk and device admission; measure actual device memory for a CUDA run,
respect the shared physical owner and retain synchronized cleanup evidence.
When a compatible reviewed owner is not available, use a recorded CPU fallback
without reporting its wall time as GPU performance.

## Exposure and interpretation

Before any pretrained forward, bind the fixed case, protocol, source, tokenizer
and checkpoint, and compare all 160 rows against the latest named public BERT
inputs and known fixed pending proposals. For full-row-prefix checks including
stripped terminal `[SEP]`, the only allowed within-base relations are each
terminal variant versus its four continued variants: 128 expected relations;
all other within-proposal and cross-bundle relations remain visible. Record
lemma and surface-frame overlap separately. These checks say nothing about
unknown, private or pretraining exposure. Have independent reviews of case
construction, controls, source, numerical handling and saved replay. Publish
positive, negative and incomplete outcomes; never change the fixed criteria
after a model output.
